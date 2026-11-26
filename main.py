import torch
from  teacher_trainer import train_teacher
from student_trainer import train_student
from tree_student_trainer import train_tree_student
import config
import numpy as np
import utils
from graph import get_dataset
import os
import json

# python main.py --dataset ogbn-products --teacher GCN --student XGBTree --training_mode T+S --runs 1 --xgb_stage_rounds 16,32,64,128,512
# python main.py --dataset citeseer --teacher GCN --student XGBTree --training_mode T+S --runs 1 --xgb_stage_rounds 16,32,64,128,512
# python main.py --dataset computers --teacher GAT --student XGBTree --training_mode T+S --runs 5 --xgb_stage_rounds 16,32,64


def main():
    # Parse command line arguments
    args = config.args
    device = torch.device('cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu')

    # Load dataset
    g = get_dataset(args.dataset)
    
    # 先训练教师；树学生蒸馏时会使用每个 run 对应的教师 logits。
    teacher_results = None
    if args.training_mode in ("T", "T+S"):
        teacher_results = train_teacher(g, args)
        # Log results
        utils.log_results(teacher_results, args, is_teacher=True)
        
    if args.training_mode in ("T+S", "S"):
        teacher_logits_list = []
        for i in range(args.runs):
            exp_name = f"{args.dataset}/{args.teacher}/RUN{i+1}"
            if not os.path.exists(f"pretrain/{exp_name}/out_teacher.npz"):
                with open("./param.json", 'r') as f:
                    default_params = json.load(f)
                    args.dropout = default_params["teacher"][args.dataset][args.teacher]["dropout"]
                    args.learning_rate = default_params["teacher"][args.dataset][args.teacher]["learning_rate"]
                    args.wd = default_params["teacher"][args.dataset][args.teacher]["wd"]
                    args.hid_dim = default_params["teacher"][args.dataset][args.teacher]["hid_dim"]
                teacher_results = train_teacher(g, args)
                
            if args.student == "XGBTree":
                # 树训练逐 run 读取 logits 路径，避免把所有节点 logits 同时放入内存。
                teacher_logits_list.append(f"pretrain/{exp_name}/out_teacher.npz")
            else:
                teacher_logits = np.load(f"pretrain/{exp_name}/out_teacher.npz")["arr_0"]
                torch_tensor = torch.from_numpy(teacher_logits).to(device)
                teacher_logits_list.append(torch_tensor)
        
        # XGBoost 和 PyTorch MLP 使用各自的训练入口。
        if args.student == "XGBTree":
            student_results = train_tree_student(g, args, teacher_logits_list)
        else:
            args.batch_size = g.num_nodes
            student_results = train_student(g, args, teacher_logits_list)
        
        # Log results
        utils.log_results(student_results, args, is_teacher=False)

if __name__ == "__main__":
    main()
