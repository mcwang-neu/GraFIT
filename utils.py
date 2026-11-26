import torch
import random
import numpy as np
import os

def log_results(results, args, is_teacher=True):
    file_path = f'./results/{args.dataset}'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    file_name = ""
    
    if is_teacher:
        file_name = os.path.join(file_path, f'{args.teacher}_None.txt')
    else:
        file_name = os.path.join(file_path, f'{args.teacher}_{args.student}.txt')
        
    with open(file_name, 'a+') as file:
        file.write(f'Accuracy: {results[0]:.2f}±{results[1]:.2f}\t')
        file.write(f'Inference Time: {results[2]:.5f}\t')
        file.write(f'Teacher: {args.teacher}\t')
        file.write(f'Student: {args.student}\t')
        file.write(f'Dataset: {args.dataset}\t')
        if not is_teacher and args.student == 'XGBTree':
            file.write(f'XGBoost Stage Rounds: {args.xgb_stage_rounds}\t')
            file.write(f'XGBoost Max Depth: {args.xgb_max_depth}\t')
            file.write(f'XGBoost Learning Rate: {args.xgb_learning_rate}\t')
            file.write(f'XGBoost Temperature: {args.xgb_temperature}\t')
            file.write(f'XGBoost Subsample: {args.xgb_subsample}\t')
            file.write(f'XGBoost Column Subsample: {args.xgb_colsample_bytree}\t')
        elif is_teacher:
            file.write(f'Hidden Dim: {args.hid_dim}\t')
            file.write(f'Dropout: {args.dropout}\t')
            file.write(f'Learning Rate: {args.learning_rate}\t')
            file.write(f'Weight Decay Rate: {args.wd}\t')
        else:
            file.write(f'Hidden Dim: {args.hid_dim}\t')
            file.write(f'Dropout: {args.dropout}\t')
            file.write(f'Learning Rate: {args.learning_rate}\t')
            file.write(f'Weight Decay Rate: {args.wd}\t')
            file.write(f'Batch Size: {args.batch_size}\t')
            file.write(f'Number of Student Blocks: {args.student_number}\t')
        file.write('\n')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_features(X, p):
    """
    Randomly masks a portion of features in each row of feature matrix X based on the probability p.

    Args:
        X: Feature matrix of shape (N, C) where N is the number of samples and C is the number of features.
        p: The proportion of features to be randomly masked, should be a value between 0 and 1.

    Returns:
        masked_X: The processed feature matrix with the same shape as X.
    """
    N, C = X.shape
    num_features_to_mask = int(C * p)
    mask = torch.rand(N, C).to(X.device)
    _, indices = mask.topk(num_features_to_mask, dim=1)
    masked_X = X.clone()
    masked_X.scatter_(1, indices, 1e-32)

    # Compute the sum of elements along each row (axis=1)
    row_sums = masked_X.sum(dim=1, keepdim=True)
    # Normalize each row by dividing by its sum
    normalized_X = masked_X / row_sums

    return normalized_X
