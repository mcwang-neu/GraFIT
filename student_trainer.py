import torch
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter
import nni
import torch.nn.functional as F
import os.path as osp
from scipy import stats
import numpy as np
from logging import Logger
import models
import config
import utils
from models import ProGMLP, MLPsBlock
import time
import os

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
        
    def __call__(self, val_loss, model=None, soft_targets=None):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # if model is not None:
            #     self.model_save(model)
            # if soft_targets is not None:
            #     self.logits_save(soft_targets)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def model_save(self, model):
        model.save(osp.join("pretrain", self.exp_name))

    def logits_save(self, soft_targets):
        np.savez(osp.join("pretrain", self.exp_name, "out_student.npz"), soft_targets)
        
    def load_best_state_dict(self, model, data):
        model.load_state_dict(self.best_state_dict)



###### Student Training ######
def train_student(dataset, args, teacher_logits=None):
    
    def adaptive_mixup_training(optimizer, g, model, lam, add_input=None):
        # model.train()
        optimizer.zero_grad()
        x = torch.cat([g.x[g.train_mask], add_input[g.train_mask]], dim=1)
        y = g.y[g.train_mask]
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        # print(f"Mixup lambda: {lam}")
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        _, logits = model(mixed_x)
        
        criterion = torch.nn.CrossEntropyLoss()
        mixup_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        # mixup_loss.backward()
        # optimizer.step()
        
        return mixup_loss 

    def train_per_epoch_student(optimizer, g, model, current_index=0, add_input=None):
        # model.train()
        # optimizer.zero_grad()
        x = torch.cat([g.x, add_input], dim=1)
        hid, logits = model(x)
        # cross-entropy loss
        out = F.log_softmax(logits, dim=1)
        classification_loss = F.nll_loss(out[g.train_mask], g.y[g.train_mask])
        
        # Distillation loss with teacher's soft labels
        distill_loss = F.kl_div(F.log_softmax(logits, dim=1), F.log_softmax(teacher_logits[run], dim=1), reduction='batchmean', log_target=True)
        
        # Weighted loss to encourage refinement
        loss = ((current_index + 1)**args.beta) * (args.alpha * distill_loss + (1 - args.alpha) * classification_loss)
        
        # loss.backward()
        # optimizer.step()
        return loss
    
    def single_run(model, g, args, run_id=1):
        exp_name = f"{args.dataset}/{args.student}/RUN{run_id}"
        # writer = SummaryWriter(logdir=osp.join("logs/", exp_name))
        
        # Adaptive strategy parameters
        initial_lam_base = config.args.initial_lam_base   # Base initial lambda
        max_lam = 0.5      # Maximum lambda
        min_lam = 0.1      # Minimum lambda
        tau = config.args.tau        # Threshold for lambda
        gamma = config.args.gamma      # Adjustment rate for lambda
        
        add_input_list = [torch.zeros(args.batch_size, args.hid_dim).to(device)]
        for i in range(args.student_number):
            initial_lam = min((initial_lam_base + i + 1) * gamma, max_lam)
            lam = initial_lam
    
            nni_params = nni.get_next_parameter()
            args.learning_rate = nni_params.get('learning_rate', args.learning_rate)
            args.batch_size = nni_params.get('batch_size', args.batch_size)

            optimizer = torch.optim.Adam(model.mlp_blocks[i].parameters(), lr=args.learning_rate)
            early_stopping = EarlyStopping(patience=args.patience, exp_name=exp_name)
            avg_loss = 1.0
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                model.train()
                train_loss = train_per_epoch_student(optimizer, g, model.mlp_blocks[i], current_index=i, add_input=add_input_list[i])
                mixup_loss = adaptive_mixup_training(optimizer, g, model.mlp_blocks[i], lam, add_input=add_input_list[i])
                loss = train_loss + args.mixup_gamma*mixup_loss
                
                loss.backward()
                optimizer.step()
                # Update moving average of loss
                avg_loss = 0.9 * avg_loss + 0.1 * mixup_loss.item()
                
                # Adjust lambda based on the loss, but considering the student model index
                if avg_loss < 0.9:  # If loss is low, increase lambda to make it harder
                    lam = min((lam + tau)*gamma + torch.rand(1).item()/100, max_lam)
                    # print(f"Increasing lambda to {lam}")
                else:  # If loss is high, decrease lambda to make it easier
                    lam = max((lam - tau)*gamma + torch.rand(1).item()/100, min_lam)
                    # print(f"Decreasing lambda to {lam}")
                # lam = min_lam + (lam - min_lam) * torch.rand(1).item()
                # Evaluate model on validation set
                val_acc, val_loss, _, pre_hid = pro_gmlp_dynamic_inference(model.mlp_blocks[i], g, mode="Val", add_input=add_input_list[i])
                # 早停目前只统计连续未提升轮数，不会缓存或恢复验证损失最低时的 MLP 权重。
                early_stopping(val_loss)

                # Log metrics to TensorBoard
                # writer.add_scalar('Loss/train', train_loss, epoch)
                # writer.add_scalar('Loss/val', val_loss, epoch)

                if early_stopping.early_stop:
                    # print(f'Early stopping triggered at epoch {epoch}')
                    break
                if epoch % 50 == 0:
                    print(f'[RUN {run_id}| Student {i + 1} | Epoch {epoch}] Train Loss: {train_loss:.4f} | Mixup Loss: {mixup_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}')
                
            #### end of training loop ####
            # Freeze the trained MLP student
            for param in model.mlp_blocks[i].parameters():
                param.requires_grad = False
            
             # Transfer parameters to the next MLP student
            if i < args.student_number - 1:
                with torch.no_grad():
                    model.mlp_blocks[i + 1].hid_mlp1.weight.copy_(model.mlp_blocks[i].hid_mlp1.weight)
                    model.mlp_blocks[i + 1].hid_mlp1.bias.copy_(model.mlp_blocks[i].hid_mlp1.bias)
                    model.mlp_blocks[i + 1].hid_mlp2.weight.copy_(model.mlp_blocks[i].hid_mlp2.weight)
                    model.mlp_blocks[i + 1].hid_mlp2.bias.copy_(model.mlp_blocks[i].hid_mlp2.bias)
                    model.mlp_blocks[i + 1].pred_mlp.weight.copy_(model.mlp_blocks[i].pred_mlp.weight)
                    model.mlp_blocks[i + 1].pred_mlp.bias.copy_(model.mlp_blocks[i].pred_mlp.bias)
                    model.mlp_blocks[i + 1].norm1.weight.copy_(model.mlp_blocks[i].norm1.weight)
                    model.mlp_blocks[i + 1].norm2.weight.copy_(model.mlp_blocks[i].norm2.weight)
            
            add_input = pre_hid.detach()
            add_input_list.append(add_input)
        if args.ee:
            ee_inference(model, g, mode="Test", add_input=add_input_list)
        # Final evaluation and logging
        intermediate_test_accuracy, intermediate_test_loss, inference_time, _ = pro_gmlp_dynamic_inference(model, g, mode="Test", add_input=add_input_list)
        # writer.add_scalar('Accuracy/test', intermediate_test_accuracy, args.epochs)
        early_stopping.model_save(model)
        # Reporting final results to NNI
        nni.report_intermediate_result(intermediate_test_accuracy)

        # Close the TensorBoard writer
        # writer.close()

        return intermediate_test_accuracy, inference_time
    
    ############# Main Loop ##############
    all_accuracies = []
    all_inference_times = []

    for run in range(args.runs):
        utils.set_seed(run * 10 + 1)

        run_teacher_logits = torch.as_tensor(teacher_logits[run], device=dataset.x.device)
        student = models.get_model(args.student, dataset.num_features, args.hid_dim,
            dataset.num_classes, block_num=args.student_number).to(device)
        run_accuracy, inference_time = single_run(student, dataset, args, run_id=run + 1)
        all_accuracies.append(run_accuracy)
        all_inference_times.append(inference_time)

    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_inference_time = np.mean(all_inference_times)
    print(mean_accuracy, std_accuracy, mean_inference_time)
    return mean_accuracy, std_accuracy, mean_inference_time

def ee_inference(model, g, mode="Val", add_input=None):
    model.eval()
    mask = g.val_mask if mode == "Val" else g.test_mask
    correct = 0
    total = 0
    T_cumulative = 0
    # for the early exit and anytime inference settings
    max_K = config.args.student_number
    for K in range(1, max_K+1):
        with torch.no_grad():
            logits_list = []
            conf_list = []
            for k in range(0, K):
                # 当前是按整个验证/测试 mask 的平均置信度决定全批次退出，不是逐节点 early-exit。
                # CUDA 计时还缺少 synchronize，因此 GPU 推理时间可能被低估。
                start_time = time.time()
                if isinstance(add_input, list):
                    x = torch.cat([g.x, add_input[k]], dim=1)
                else:
                    x = torch.cat([g.x, add_input], dim=1)
                hid, logits = model.inference(x, index=k)
                conf, _ = F.softmax(logits, dim=1).max(1)
                conf = conf[mask].mean()
                # Update cumulative execution time
                T_cumulative += time.time() - start_time
                logits_list.append(logits)
                conf_list.append(conf)
                
                # Check if confidence exceeds the threshold or time budget is nearly exhausted
                if conf >= config.args.conf_tau or T_cumulative >= config.args.T_budget:
                    break  # Early stop with current prediction
            # print(f"Triggered at student {k+1} with confidence {conf:.4f}")
            # Calculate the final prediction
            logits = torch.zeros(g.num_nodes, g.num_classes).to(device)
            conf_list = torch.stack(conf_list)
            conf_weights = F.softmax(conf_list, dim=0)
            for i in range(len(logits_list)):
                logits += conf_weights[i] * logits_list[i]
            out = logits
            out = out[mask]
            y = g.y[mask]
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            loss = F.nll_loss(out.log_softmax(1), y)

            accuracy = correct / total
            print(accuracy*100)
        
    return accuracy * 100, loss, T_cumulative, hid

def pro_gmlp_dynamic_inference(model, g, mode="Val", add_input=None):
    model.eval()
    mask = g.val_mask if mode == "Val" else g.test_mask
    correct = 0
    total = 0
    T_cumulative = 0
    # for the early exit and anytime inference settings
    K = config.args.student_number
    with torch.no_grad():
        if isinstance(model, ProGMLP):
            logits_list = []
            conf_list = []
            for k in range(0, K):
                # 这里用整个评估 mask 的平均置信度统一决定是否继续，而非逐节点退出。
                # 若在 CUDA 上计时，缺少 synchronize 会低估实际耗时。
                start_time = time.time()
                if isinstance(add_input, list):
                    x = torch.cat([g.x, add_input[k]], dim=1)
                else:
                    x = torch.cat([g.x, add_input], dim=1)
                hid, logits = model.inference(x, index=k)
                conf, _ = F.softmax(logits, dim=1).max(1)
                conf = conf[mask].mean()
                # Update cumulative execution time
                T_cumulative += time.time() - start_time
                logits_list.append(logits)
                conf_list.append(conf)
                
                # # Check if confidence exceeds the threshold or time budget is nearly exhausted
                if conf >= config.args.conf_tau:
                    break  # Early stop with current prediction
            # print(f"Triggered at student {k+1} with confidence {conf:.4f}")
            # Calculate the final prediction
            logits = torch.zeros(g.num_nodes, g.num_classes).to(device)
            conf_list = torch.stack(conf_list)
            conf_weights = F.softmax(conf_list, dim=0)
            for i in range(len(logits_list)):
                logits += conf_weights[i] * logits_list[i]
        else:
            x = torch.cat([g.x, add_input], dim=1)
            hid, logits = model.inference(x)
        
        out = logits
        out = out[mask]
        y = g.y[mask]
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        loss = F.nll_loss(out.log_softmax(1), y)

        accuracy = correct / total
        
    return accuracy * 100, loss, T_cumulative, hid


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def perform_p_test(accuracies1, accuracies2):
    return stats.ttest_ind(accuracies1, accuracies2)
