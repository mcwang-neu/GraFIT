import csv
import os
import shutil
import time

import numpy as np
import torch

import utils


def _as_numpy(value, dtype=None):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def _reshape_margins(raw_predictions, num_rows, num_classes):
    """Normalize XGBoost's version-dependent multiclass output layout."""
    raw_predictions = np.asarray(raw_predictions)
    if raw_predictions.shape == (num_rows, num_classes):
        return raw_predictions
    if raw_predictions.shape == (num_classes, num_rows):
        return raw_predictions.T
    if raw_predictions.size == num_rows * num_classes:
        return raw_predictions.reshape(num_rows, num_classes)
    raise ValueError(
        "Unexpected XGBoost multiclass prediction shape: "
        f"{raw_predictions.shape}; expected {num_rows} x {num_classes}."
    )


def _classification_metrics(probabilities, labels, mask):
    selected_probabilities = np.clip(probabilities[mask], 1e-12, 1.0)
    selected_labels = labels[mask]
    row_ids = np.arange(selected_labels.shape[0])
    loss = -np.log(selected_probabilities[row_ids, selected_labels]).mean()
    accuracy = (selected_probabilities.argmax(axis=1) == selected_labels).mean()
    return float(accuracy * 100.0), float(loss)


def _get_stage_rounds(args):
    try:
        rounds = sorted(set(int(value.strip()) for value in args.xgb_stage_rounds.split(",") if value.strip()))
    except ValueError as exc:
        raise ValueError("--xgb_stage_rounds must be comma-separated positive integers, e.g. 16,32,64,128") from exc
    if not rounds or rounds[0] <= 0:
        raise ValueError("--xgb_stage_rounds must contain positive boosting-round counts")
    return rounds


def _make_distillation_objective(teacher_probabilities, labels, train_mask, alpha, temperature):
    """Combine teacher KL gradients on all nodes with hard-label CE on train nodes.

    The full-node teacher term matches the repository's transductive distillation
    setup. Ground-truth labels contribute only for nodes selected by train_mask.
    """
    num_nodes, num_classes = teacher_probabilities.shape
    train_ids = np.flatnonzero(train_mask)
    train_targets = np.eye(num_classes, dtype=np.float32)[labels[train_ids]]
    distill_scale = alpha
    supervised_scale = (1.0 - alpha) * num_nodes / float(max(len(train_ids), 1))

    def objective(predictions, _dtrain):
        predictions = np.asarray(predictions)
        original_shape = predictions.shape
        margins = _reshape_margins(predictions, num_nodes, num_classes)

        teacher_student_prob = _softmax(margins / temperature)
        hard_student_prob = _softmax(margins)

        grad = (distill_scale / temperature) * (teacher_student_prob - teacher_probabilities)
        hess = (distill_scale / (temperature ** 2)) * np.maximum(
            teacher_student_prob * (1.0 - teacher_student_prob), 1e-6
        )

        if len(train_ids):
            grad[train_ids] += supervised_scale * (hard_student_prob[train_ids] - train_targets)
            hess[train_ids] += supervised_scale * np.maximum(
                hard_student_prob[train_ids] * (1.0 - hard_student_prob[train_ids]), 1e-6
            )

        grad = grad.astype(np.float32, copy=False)
        hess = hess.astype(np.float32, copy=False)
        if len(original_shape) == 2 and original_shape == (num_classes, num_nodes):
            return grad.T, hess.T
        return grad.reshape(original_shape), hess.reshape(original_shape)

    return objective


def train_tree_student(dataset, args, teacher_logits_list):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("XGBoost student requested. Install the pinned dependency with `pip install -r requirements.txt`.") from exc

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1 for the XGBoost distillation objective")
    if args.xgb_temperature <= 0:
        raise ValueError("--xgb_temperature must be positive")

    stage_rounds = _get_stage_rounds(args)
    features = _as_numpy(dataset.x, np.float32)
    labels = _as_numpy(dataset.y, np.int64).reshape(-1)
    train_mask = _as_numpy(dataset.train_mask, bool).reshape(-1)
    val_mask = _as_numpy(dataset.val_mask, bool).reshape(-1)
    test_mask = _as_numpy(dataset.test_mask, bool).reshape(-1)
    num_nodes = features.shape[0]
    num_classes = int(dataset.num_classes)

    if features.shape[0] != labels.shape[0] or len(teacher_logits_list) != args.runs:
        raise ValueError("Feature/label sizes or teacher-logit run count do not match the dataset and --runs")
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("The dataset must have non-empty train, validation, and test masks")

    dmatrix = xgb.DMatrix(features)
    all_stage_rows = []
    best_test_accuracies = []
    best_inference_times = []

    model_root = os.path.join("pretrain", args.dataset, args.student)
    result_root = os.path.join("results", args.dataset)
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(result_root, exist_ok=True)
    stage_csv = os.path.join(result_root, f"{args.teacher}_{args.student}_stages.csv")

    for run_index in range(args.runs):
        seed = run_index * 10 + 1
        utils.set_seed(seed)
        teacher_source = teacher_logits_list[run_index]
        if isinstance(teacher_source, (str, os.PathLike)):
            with np.load(teacher_source) as teacher_archive:
                teacher_logits = _as_numpy(teacher_archive["arr_0"], np.float32)
        else:
            teacher_logits = _as_numpy(teacher_source, np.float32)
        if teacher_logits.shape != (num_nodes, num_classes):
            raise ValueError(
                f"Teacher logits for run {run_index + 1} have shape {teacher_logits.shape}; "
                f"expected {(num_nodes, num_classes)}"
            )

        teacher_probabilities = _softmax(teacher_logits / args.xgb_temperature).astype(np.float32)
        objective = _make_distillation_objective(
            teacher_probabilities, labels, train_mask, args.alpha, args.xgb_temperature
        )
        params = {
            "objective": "multi:softprob",
            "num_class": num_classes,
            "disable_default_eval_metric": 1,
            "tree_method": "hist",
            "eta": args.xgb_learning_rate,
            "max_depth": args.xgb_max_depth,
            "min_child_weight": args.xgb_min_child_weight,
            "subsample": args.xgb_subsample,
            "colsample_bytree": args.xgb_colsample_bytree,
            "lambda": args.xgb_reg_lambda,
            "alpha": args.xgb_reg_alpha,
            "nthread": args.xgb_n_jobs,
            "seed": seed,
        }

        start_train = time.perf_counter()
        booster = xgb.train(
            params=params,
            dtrain=dmatrix,
            num_boost_round=stage_rounds[-1],
            obj=objective,
            verbose_eval=False,
        )
        train_seconds = time.perf_counter() - start_train

        run_dir = os.path.join(model_root, f"RUN{run_index + 1}")
        os.makedirs(run_dir, exist_ok=True)
        run_stage_rows = []
        for rounds in stage_rounds:
            start_inference = time.perf_counter()
            margins = booster.predict(
                dmatrix,
                output_margin=True,
                iteration_range=(0, rounds),
            )
            probabilities = _softmax(_reshape_margins(margins, num_nodes, num_classes))
            inference_seconds = time.perf_counter() - start_inference

            val_accuracy, val_loss = _classification_metrics(probabilities, labels, val_mask)
            test_accuracy, test_loss = _classification_metrics(probabilities, labels, test_mask)

            stage_path = os.path.join(run_dir, f"stage_{rounds}.json")
            booster[:rounds].save_model(stage_path)
            row = {
                "run": run_index + 1,
                "rounds": rounds,
                "val_accuracy": val_accuracy,
                "val_logloss": val_loss,
                "test_accuracy": test_accuracy,
                "test_logloss": test_loss,
                "inference_seconds": inference_seconds,
                "train_seconds": train_seconds,
                "model_size_bytes": os.path.getsize(stage_path),
                "best_by_validation": False,
            }
            run_stage_rows.append(row)
            print(
                f"[RUN {run_index + 1} | XGB stage {rounds}] "
                f"Val Acc: {val_accuracy:.2f}% | Test Acc: {test_accuracy:.2f}% | "
                f"Inference: {inference_seconds:.4f}s"
            )

        best_row = min(run_stage_rows, key=lambda row: row["val_logloss"])
        best_row["best_by_validation"] = True
        shutil.copy2(
            os.path.join(run_dir, f"stage_{best_row['rounds']}.json"),
            os.path.join(run_dir, "best.json"),
        )
        best_test_accuracies.append(best_row["test_accuracy"])
        best_inference_times.append(best_row["inference_seconds"])
        all_stage_rows.extend(run_stage_rows)

    fieldnames = list(all_stage_rows[0].keys())
    write_header = not os.path.exists(stage_csv) or os.path.getsize(stage_csv) == 0
    with open(stage_csv, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_stage_rows)

    mean_accuracy = float(np.mean(best_test_accuracies))
    std_accuracy = float(np.std(best_test_accuracies))
    mean_inference_time = float(np.mean(best_inference_times))
    print(
        "\n========== XGBoost Student Results (best validation prefix) ==========\n"
        f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}%\n"
        f"Mean inference time: {mean_inference_time:.4f}s\n"
        f"Stage metrics: {stage_csv}\n"
        "====================================================================="
    )
    return mean_accuracy, std_accuracy, mean_inference_time
