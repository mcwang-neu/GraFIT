"""One-factor-at-a-time sensitivity analysis for the Graph-to-Tree student.

Example:
    python para.py --dataset cora --teacher GCN --runs 5

The script uses existing teacher logits, keeps the data split fixed, and writes
its results to a new timestamped CSV under results/parameter_sensitivity/.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import sys

def _parse_values(value, cast, name):
    try:
        values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {name} values: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def _parse_args():
    parser = argparse.ArgumentParser(description="Graph-to-Tree parameter sensitivity analysis")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--teacher", default="GCN")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--xgb-stage-rounds", default="16,32,64,128")
    parser.add_argument("--alpha-values", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--temperature-values", default="0.5,1,2,4")
    parser.add_argument("--max-depth-values", default="5,7,9,12,18,24")
    parser.add_argument("--learning-rate-values", "--xgboost-learning-rate-values",
                        dest="learning_rate_values", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-subsample", type=float, default=1.0)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")

    alpha_values = _parse_values(args.alpha_values, float, "alpha")
    temperature_values = _parse_values(args.temperature_values, float, "temperature")
    depth_values = _parse_values(args.max_depth_values, int, "max_depth")
    learning_rate_values = _parse_values(args.learning_rate_values, float, "learning_rate")
    try:
        stage_rounds = sorted(set(int(item.strip()) for item in args.xgb_stage_rounds.split(",") if item.strip()))
    except ValueError as exc:
        raise SystemExit("--xgb-stage-rounds must be comma-separated positive integers") from exc
    if not stage_rounds or stage_rounds[0] <= 0:
        raise SystemExit("--xgb-stage-rounds must contain positive integers")
    if any(not 0.0 <= value <= 1.0 for value in alpha_values):
        raise SystemExit("All alpha values must be between 0 and 1")
    if any(value <= 0 for value in temperature_values + learning_rate_values):
        raise SystemExit("Temperature and learning-rate values must be positive")
    if any(value <= 0 for value in depth_values):
        raise SystemExit("All max-depth values must be positive")

    import numpy as np

    # Existing modules parse command-line arguments at import time; clear argv
    # so this script's own options are not passed into that legacy parser.
    sys.argv = [sys.argv[0]]
    from graph import get_dataset
    from tree_student_trainer import _classification_metrics, _make_distillation_objective, _reshape_margins, _softmax
    import xgboost as xgb

    dataset = get_dataset(args.dataset)
    features = dataset.x.detach().cpu().numpy().astype(np.float32, copy=False)
    labels = dataset.y.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    train_mask = dataset.train_mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    val_mask = dataset.val_mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    test_mask = dataset.test_mask.detach().cpu().numpy().astype(bool, copy=False).reshape(-1)
    num_nodes = features.shape[0]
    num_classes = int(dataset.num_classes)
    dmatrix = xgb.DMatrix(features)

    teacher_paths = [Path("pretrain") / args.dataset / args.teacher / f"RUN{i + 1}" / "out_teacher.npz"
                     for i in range(args.runs)]
    missing = [str(path) for path in teacher_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing pretrained teacher logits; train the teacher first or reduce --runs. "
            + ", ".join(missing)
        )
    teacher_logits = []
    for path in teacher_paths:
        with np.load(path) as archive:
            logits = np.asarray(archive["arr_0"], dtype=np.float32)
        if logits.shape != (num_nodes, num_classes):
            raise ValueError(f"Unexpected teacher-logit shape in {path}: {logits.shape}")
        teacher_logits.append(logits)

    analyses = [
        ("alpha", alpha_values),
        ("xgb_temperature", temperature_values),
        ("xgb_max_depth", depth_values),
        ("xgb_learning_rate", learning_rate_values),
    ]
    base = {
        "alpha": 0.5,
        "xgb_temperature": 1.0,
        "xgb_max_depth": 12,
        "xgb_learning_rate": 0.1,
    }
    records = []

    for parameter, values in analyses:
        for value in values:
            setting = dict(base)
            setting[parameter] = value
            run_val_losses = []
            run_val_accuracies = []
            run_test_accuracies = []
            selected_rounds = []

            for run_index, logits in enumerate(teacher_logits):
                seed = run_index * 10 + 1
                np.random.seed(seed)
                teacher_probs = _softmax(logits / setting["xgb_temperature"]).astype(np.float32)
                objective = _make_distillation_objective(
                    teacher_probs, labels, train_mask, setting["alpha"], setting["xgb_temperature"]
                )
                params = {
                    "objective": "multi:softprob",
                    "num_class": num_classes,
                    "disable_default_eval_metric": 1,
                    "tree_method": "hist",
                    "eta": setting["xgb_learning_rate"],
                    "max_depth": setting["xgb_max_depth"],
                    "min_child_weight": args.xgb_min_child_weight,
                    "subsample": args.xgb_subsample,
                    "colsample_bytree": args.xgb_colsample_bytree,
                    "lambda": args.xgb_reg_lambda,
                    "alpha": args.xgb_reg_alpha,
                    "nthread": -1,
                    "seed": seed,
                    "verbosity": 0,
                }
                booster = xgb.train(
                    params=params,
                    dtrain=dmatrix,
                    num_boost_round=stage_rounds[-1],
                    obj=objective,
                    verbose_eval=False,
                )

                stage_metrics = []
                for rounds in stage_rounds:
                    margins = booster.predict(dmatrix, output_margin=True, iteration_range=(0, rounds))
                    probabilities = _softmax(_reshape_margins(margins, num_nodes, num_classes))
                    val_accuracy, val_loss = _classification_metrics(probabilities, labels, val_mask)
                    test_accuracy, _ = _classification_metrics(probabilities, labels, test_mask)
                    stage_metrics.append((val_loss, val_accuracy, test_accuracy, rounds))
                best = min(stage_metrics, key=lambda item: item[0])
                run_val_losses.append(best[0])
                run_val_accuracies.append(best[1])
                run_test_accuracies.append(best[2])
                selected_rounds.append(best[3])

            records.append({
                "parameter": parameter,
                "value": value,
                "val_logloss_mean": float(np.mean(run_val_losses)),
                "val_logloss_std": float(np.std(run_val_losses)),
                "val_accuracy_mean": float(np.mean(run_val_accuracies)),
                "test_accuracy_mean": float(np.mean(run_test_accuracies)),
                "selected_rounds_mean": float(np.mean(selected_rounds)),
                "runs": args.runs,
            })

    output_dir = Path("results") / "parameter_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{args.dataset}_{args.teacher}_{timestamp}.csv"
    fields = list(records[0].keys())
    with output_path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summaries = []
    for parameter, _ in analyses:
        group = [row for row in records if row["parameter"] == parameter]
        best = min(group, key=lambda row: row["val_logloss_mean"])
        summaries.append(f"{parameter}: best={best['value']} (val log-loss={best['val_logloss_mean']:.4f})")
    print(f"Sensitivity results: {output_path}")
    print("; ".join(summaries))


if __name__ == "__main__":
    main()
