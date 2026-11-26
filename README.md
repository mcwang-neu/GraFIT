# GraFIT

## This is an anonymous code repository for GraFIT. 

To reproduce the results in our paper, you need to `pip install -r requirements.txt```. And then: 

```

python main.py --dataset pubmed --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset cora --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset coauthor --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset physics --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset photo --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset computers --teacher GCN --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512

python main.py --dataset pubmed --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset cora --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset coauthor --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset physics --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset photo --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512
python main.py --dataset computers --teacher GAT --student XGBTree --training_mode T+S --runs 10 --xgb_stage_rounds 16,32,64,128,512

```
