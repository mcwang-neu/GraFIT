import torch
from torch_geometric.data import DataLoader
# from tensorboardX import SummaryWriter
# import nni
import torch.nn.functional as F
import os.path as osp
from scipy import stats
import numpy as np
from logging import Logger
import utils
import models
import config
import time

device = config.device

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, exp_name=""):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_dict = None
        self.exp_name = exp_name
        
    def __call__(self, val_loss, model=None, logits=None):
        if self.best_loss is None:
            # 首次只记录损失、不保存模型/logits；若之后没有更优轮次，缓存文件不会生成。
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                model.save(osp.join("pretrain", self.exp_name))
            if logits is not None:
                np.savez(osp.join("pretrain", self.exp_name, "out_teacher.npz"), logits)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_state_dict(self, model, data):
        model.load_state_dict(self.best_state_dict)


def bacth_train_per_epoch(optimizer, g, model, args):
    train_loader = DataLoader(g.dataset, batch_size=args.batch_size, shuffle=True)
    total_loss = 0
    for batch_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch)
        loss = F.nll_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(train_loader)

    return total_loss

###### Teacher Training ######
def train_teacher(dataset, args):
    def train_per_epoch(optimizer, g, model):
        optimizer.zero_grad()
        logits = model(g)
        out = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(out[g.train_mask], g.y[g.train_mask])
        loss.backward()
        optimizer.step()
        return loss, logits

    def single_run(model, g, args, run_id=1):
        # Identify this run
        exp_name = f"{args.dataset}/{model.name}/RUN{run_id}"
        # Initialize TensorBoardX writer
        # writer = SummaryWriter(logdir=osp.join("logs/", exp_name))

        # Get hyperparameters from NNI
        # nni_params = nni.get_next_parameter()
        # args.learning_rate = nni_params.get('learning_rate', args.learning_rate)
        # args.batch_size = nni_params.get('batch_size', args.batch_size)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        early_stopping = EarlyStopping(patience=args.patience, exp_name=exp_name)

        for epoch in range(args.epochs):

            train_loss, logits = train_per_epoch(optimizer, g, model)

            val_acc, val_loss, _ = evaluate_model(model, g, mode="Val")

            # 注意：logits 在 optimizer.step 前计算，而 model 此时已更新；保存的 logits 与模型参数可能差一个更新步。
            early_stopping(val_loss, model, logits.detach().cpu())

            # Log metrics to TensorBoard
            # writer.add_scalar('Loss/train', train_loss, epoch)
            # writer.add_scalar('Loss/val', val_loss, epoch)

            if early_stopping.early_stop:
                print(f'Early stopping triggered at epoch {epoch}')
                break

            if epoch % 50 == 0:
                print(f'[RUN {run_id} | Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}')

        # 此处使用停止时的最后一轮模型评估；没有从早停保存文件恢复验证集最优模型。
        intermediate_test_accuracy, intermediate_test_loss, inference_time = evaluate_model(model, g, mode="Test")
        # writer.add_scalar('Accuracy/test', intermediate_test_loss, args.epochs)

        # Reporting final results to NNI
        # nni.report_intermediate_result(intermediate_test_accuracy)

        # Close the TensorBoard writer
        # writer.close()

        return intermediate_test_accuracy, inference_time

    all_accuracies = []
    all_inference_times = []

    for run in range(args.runs):
        utils.set_seed(run * 10 + 1)

        teacher = models.get_model(
            args.teacher,
            dataset.num_features,
            args.hid_dim,
            dataset.num_classes
        ).to(device)

        run_accuracy, inference_time = single_run(
            teacher, dataset, args, run_id=run + 1
        )

        all_accuracies.append(run_accuracy)
        all_inference_times.append(inference_time)

        print(
            f"Run {run + 1}/{args.runs} | "
            f"Test Accuracy: {run_accuracy:.2f}% | "
            f"Inference Time: {inference_time:.4f}s"
        )

    # Calculate mean and standard deviation
    mean_inference_time = np.mean(all_inference_times)
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)

    print("\n========== Teacher Training Results ==========")
    print(f"Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"Std Accuracy: {std_accuracy:.2f}%")
    print(f"Mean Inference Time: {mean_inference_time:.4f}s")
    print("==============================================")

    return mean_accuracy, std_accuracy, mean_inference_time


def evaluate_model(model, g, mode="Val"):
    model.eval()
    mask = g.val_mask if mode == "Val" else g.test_mask
    correct = 0
    total = 0
    with torch.no_grad():
        start_time = time.time()
        out = model(g)
        preds = out
        out = out[mask]
        y = g.y[mask]
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        loss = F.nll_loss(out.log_softmax(1), y)
        end_time = time.time()
    accuracy = correct / total
    return accuracy * 100, loss, end_time-start_time


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def perform_p_test(accuracies1, accuracies2):
    return stats.ttest_ind(accuracies1, accuracies2)
