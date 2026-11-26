import os
import torch.nn as nn
import torch
import torch_geometric.nn as geom_nn
import os.path as osp
import torch.nn.functional as F
import config

######### For BaseModel ##########
class BaseModel(torch.nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, num_layers=2):
        super().__init__()
        self.embed = None

    def update_embed(self, embed):
        self.embed = embed

    def get_embed(self):
        if self.embed is not None:
            return self.embed

    def save(self, filename):
        if not osp.exists(filename):
            os.makedirs(filename)
        torch.save(self.state_dict(), osp.join(filename, "model.pth"))

    def inference(self, x):
        return self.inference(x)
    
    @classmethod
    def load(cls, filename, *args, **kwargs):
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(filename))
        model.eval()
        return model

class GCNModel(BaseModel):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GCNModel, self).__init__(in_dim, hid_dim, out_dim)
        self.conv1 = geom_nn.GCNConv(in_dim, hid_dim)
        self.conv2 = geom_nn.GCNConv(hid_dim, hid_dim)
        self.conv3 = geom_nn.GCNConv(hid_dim, out_dim)

    @property
    def name(self):
        return "GCN"
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        self.update_embed(x)
        x = x.relu()
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self.update_embed(x)
        x = x.relu()
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


class GATModel(BaseModel):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(GATModel, self).__init__(in_dim, hid_dim, out_dim)
        heads = 2
        self.conv1 = geom_nn.GATConv(in_dim, hid_dim, heads=heads)
        self.conv2 = geom_nn.GATConv(hid_dim*heads, hid_dim, heads=heads)
        self.conv3 = geom_nn.GATConv(hid_dim*heads, out_dim, heads=1)

    @property
    def name(self):
        return "GAT"
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        self.update_embed(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self.update_embed(x)
        x = F.elu(x)
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        
        return x
    

class SAGEModel(BaseModel):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(SAGEModel, self).__init__(in_dim, hid_dim, out_dim)
        self.conv1 = geom_nn.SAGEConv(in_dim, hid_dim)
        self.conv2 = geom_nn.SAGEConv(hid_dim, hid_dim)
        self.conv3 = geom_nn.SAGEConv(hid_dim, out_dim)

    @property
    def name(self):
        return "SAGE"
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        self.update_embed(x)
        x = x.relu()
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        self.update_embed(x)
        x = x.relu()
        
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

######## For Designted ProGMLP ##########
class ProGMLP(BaseModel):
    def __init__(self, in_dim, hid_dim, out_dim=1, block_num=2):
        super(ProGMLP, self).__init__(in_dim, hid_dim, out_dim)
        self.mlp_blocks = nn.ModuleList()
        for _ in range(block_num):
            self.mlp_blocks.append(MLPsBlock(in_dim+hid_dim, hid_dim, out_dim))
    
    def forward(self, x, pre_hidden=None, index=0):  
        x = F.dropout(x, p=config.args.dropout, training=self.training)
        hid, logits = self.mlp_blocks[index](x)
        
        return hid, logits
    
    def inference(self, x, index=-1):  
        hid, logits = self.mlp_blocks[index].inference(x)
        
        return hid, logits
            
    
class MLPsBlock(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLPsBlock, self).__init__()

        self.hid_mlp1 = nn.Linear(in_dim, hid_dim)
        self.hid_mlp2 = nn.Linear(hid_dim, hid_dim)
        self.pred_mlp = nn.Linear(hid_dim, out_dim)
            
        self.norm1 = nn.BatchNorm1d(hid_dim)
        self.norm2 = nn.BatchNorm1d(hid_dim)

    def forward(self, x):
        hid = self.hid_mlp1(x)
        hid = self.norm1(hid)
        hid = F.relu(hid)
        hid = self.norm2(hid)
        hid = F.relu(hid)
        logits = self.pred_mlp(hid)
    
        return hid, logits
    
    def inference(self, x):  
        hid = self.hid_mlp1(x)
        hid = self.norm1(hid)
        hid = F.relu(hid)
        hid = self.norm2(hid)
        hid = F.relu(hid)
        logits = self.pred_mlp(hid)
        
        return hid, logits

######## For utils ##########
def get_model(model_type, in_dim, hid_dim, out_dim, block_num=1):
    if model_type == 'GCN':
        return GCNModel(in_dim, hid_dim, out_dim)
    elif model_type == 'GAT':
        return GATModel(in_dim, hid_dim, out_dim)
    elif model_type == 'SAGE':
        return SAGEModel(in_dim, hid_dim, out_dim)
    elif model_type == 'ProGMLP':
        return ProGMLP(in_dim, hid_dim, out_dim, block_num=block_num)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
