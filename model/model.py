import torch.nn as nn
from utils.log import *

class Model(nn.Module):
    def __init__(self, gnn_layers: int=1):
        super(Model, self).__init__()
        
        self.gnn_layers = gnn_layers
        
    def __str__(self):
        return light_blue(f"Model with {self.gnn_layers} GNN layers")