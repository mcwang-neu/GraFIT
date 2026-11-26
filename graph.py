from torch_geometric.datasets import Planetoid, Coauthor, Amazon, CitationFull
import torch
import torch_geometric.transforms as T
from config import device as global_device
from torch_sparse import SparseTensor
import random
from torch_geometric.utils import degree
import torch.nn.functional as F
from typing import List, Tuple

ROOT = "/home/xxx/data"

class Graph:
    def __init__(self, name: str, x: torch.FloatTensor, y: torch.LongTensor, 
                 edge_index: torch.LongTensor, num_features: int, num_nodes: int, num_classes: int, 
                 train_mask: torch.BoolTensor, val_mask: torch.BoolTensor, test_mask: torch.BoolTensor, 
                 unlabeled_mask: torch.BoolTensor):
        self._name = name
        self._x = x
        self._y = y
        self._edge_index = edge_index
        self._num_features = num_features
        self._num_nodes = num_nodes
        self._num_classes = num_classes
        self._train_mask = train_mask
        self._val_mask = val_mask
        self._test_mask = test_mask
        self._unlabeled_mask = unlabeled_mask
   

    @property
    def name(self):
        return self._name

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def edge_index(self):
        return self._edge_index

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_mask(self):
        return self._train_mask

    @property
    def val_mask(self):
        return self._val_mask

    @property
    def test_mask(self):
        return self._test_mask

    @property
    def unlabeled_mask(self):
        return self._unlabeled_mask
   
    def to(self, device):
        self._x = self._x.to(device)
        self._y = self._y.to(device)
        self._edge_index = self._edge_index.to(device)
        self._train_mask = self._train_mask.to(device)
        self._val_mask = self._val_mask.to(device)
        self._test_mask = self._test_mask.to(device)
        self._unlabeled_mask = self._unlabeled_mask.to(device)
      
    def __repr__(self):
        return f"Graph(name={self.name}, num_nodes={self.num_nodes}, num_features={self.num_features}, num_classes={self.num_classes})"
 

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_splits_CS(y, num_classes, percls_trn=20, percls_val=30):
    labels = torch.clone(y)
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(rest_index, size=num_nodes)

    return train_mask, val_mask, test_mask

def random_splits_planetoid(y, num_classes, percls_trn=20, val_size=500):
    labels = torch.clone(y)
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(rest_index[:val_size], size=num_nodes)
    test_mask = index_to_mask(rest_index[val_size:], size=num_nodes)

    return train_mask, val_mask, test_mask

def random_splits_CS(y, num_classes, percls_trn=20, percls_val=30):
    labels = torch.clone(y)
    num_nodes = labels.shape[0]
    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    val_index = torch.cat([i[percls_trn:percls_trn + percls_val] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn + percls_val:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(rest_index, size=num_nodes)

    return train_mask, val_mask, test_mask
    
def get_dataset(name: str, num_per_class: int=-1) -> Graph:
    if name in ["cora", "citeseer", "pubmed"]:
        transform = T.NormalizeFeatures()
        dataset = Planetoid(ROOT, name, transform=transform)
        data = dataset[0]
        x, y, edge_index, num_features, num_nodes, num_classes, train_mask, val_mask, test_mask, unlabeled_mask = data.x, data.y, data.edge_index, dataset.num_features, data.num_nodes, dataset.num_classes, data.train_mask, data.val_mask, data.test_mask, ~data.train_mask
        if num_per_class > 0:
            train_mask, val_mask, test_mask = random_splits_planetoid(y, num_classes, percls_trn=num_per_class)
            unlabeled_mask = ~train_mask
        g = Graph(name, x, y, edge_index, num_features, num_nodes, num_classes, train_mask, val_mask, test_mask, unlabeled_mask)
        g.to(global_device)
        
    elif name in ["coauthor", "physics"]:
        transform = T.NormalizeFeatures()
        name = "CS" if name != "physics" else "physics"
        dataset = Coauthor(ROOT, name, transform=transform)
        data = dataset[0]
        x, y, edge_index, num_features, num_nodes, num_classes = data.x, data.y, data.edge_index, dataset.num_features, data.num_nodes, dataset.num_classes
        if num_per_class > 0:
            train_mask, val_mask, test_mask = random_splits_CS(y, num_classes, percls_trn=num_per_class)
            unlabeled_mask = ~train_mask
        else:
            train_mask, val_mask, test_mask = random_splits_CS(y, num_classes)
            unlabeled_mask = ~train_mask
        g = Graph(name, x, y, edge_index, num_features, num_nodes, num_classes, train_mask, val_mask, test_mask, unlabeled_mask)
        g.to(global_device)


    elif name in ["photo", "computers"]:
        transform = T.NormalizeFeatures()

        # PyG 中的数据集名称分别是 Photo 和 Computers
        dataset_name = "Photo" if name == "photo" else "Computers"

        dataset = Amazon(
            ROOT,
            dataset_name,
            transform=transform
        )
        data = dataset[0]

        x = data.x
        y = data.y
        edge_index = data.edge_index
        num_features = dataset.num_features
        num_nodes = data.num_nodes
        num_classes = dataset.num_classes

        if num_per_class > 0:
            train_mask, val_mask, test_mask = random_splits_CS(y, num_classes, percls_trn=num_per_class)
        else:
            train_mask, val_mask, test_mask = random_splits_CS(y, num_classes)

        unlabeled_mask = ~train_mask
        g = Graph(name, x, y, edge_index, num_features, num_nodes, num_classes, train_mask, val_mask, test_mask,unlabeled_mask)
        g.to(global_device)

    elif name in ['ogbn-arxiv', 'ogbn-products']:
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name=name, root=ROOT, transform=T.ToSparseTensor())
        data = dataset[0]
        x, y, num_features, num_nodes, num_classes = data.x, data.y.squeeze(1), dataset.num_features, data.num_nodes, dataset.num_classes
        row, col, _ = data.adj_t.to_symmetric().coo()  # data is torch_geometric.data.data.Data
        edge_index = torch.stack([row, col], axis=0)
        split_idx = dataset.get_idx_split()
        train_mask, val_mask, test_mask = index_to_mask(split_idx['train'], num_nodes), index_to_mask(split_idx['valid'], num_nodes), index_to_mask(split_idx['test'], num_nodes)
        unlabeled_mask = ~train_mask
        g = Graph(name, x, y, edge_index, num_features, num_nodes, num_classes, train_mask, val_mask, test_mask, unlabeled_mask)
        g.to(global_device)
        
    return g
