import torch
import torch.nn as nn
from torch.nn.functional import dropout

from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

from utils.log import *
from utils.const import GCN_MODEL_SETTINGS

class GCN(nn.Module):
    def __init__(self, 
                 embed_dims: list=[1, 1]):
        
        super(GCN, self).__init__()

        self.num_gcn_layers = len(embed_dims) - 1
        self.embed_dims     = embed_dims
        
        self.setup_layers()

    def setup_layers(self):
        # Setup the GCN layers
        gcn_layers = []
        for layer_idx in range(self.num_gcn_layers):
            gcn_layers.append(GCNConv(self.embed_dims[layer_idx], self.embed_dims[layer_idx + 1]))

            if layer_idx != self.num_gcn_layers - 1:
                gcn_layers.append(nn.ReLU())

        self.gcn = nn.ModuleList(gcn_layers)
        
    def forward(self, X, edges):
        for layer in self.gcn:
            if isinstance(layer, GCNConv):
                X = layer(X, edges)
            else:
                X = layer(X)

        return X

class MLP(nn.Module):
    def __init__(self, 
                 embed_dims: list=[1, 1]):
        
        super(MLP, self).__init__()

        self.num_fc_layers = len(embed_dims) - 1
        self.embed_dims    = embed_dims
        
        self.setup_layers()

    def setup_layers(self):
        # Setup the GCN layers
        mlp_layers = []
        for layer_idx in range(self.num_fc_layers):
            mlp_layers.append(nn.Linear(self.embed_dims[layer_idx], self.embed_dims[layer_idx + 1]))
            
            if layer_idx != self.num_fc_layers - 1:
                mlp_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, X):
        X = self.mlp(X)
        X = X.sigmoid()
        
        return X

class GCNModel(nn.Module):
    def __init__(self, settings):
        super(GCNModel, self).__init__()

        gcn_embed_dims, mlp_embed_dims = (settings[key] for key in GCN_MODEL_SETTINGS)

        # Initialize layers
        self.gcn = GCN(gcn_embed_dims)
        self.mlp = MLP(mlp_embed_dims)
    
        # Keep the data
        self.gcn_embed_dims = gcn_embed_dims
        self.mlp_embed_dims = mlp_embed_dims
        
        # Port the model to adequate device (preferably CUDA, otherwise CPU)
        self.port_to_device()

    def port_to_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.device = device
        
        print_done("Model ported to device: " + str(device))

    def forward(self, X, edge_index, batch):
        # Infer embeddings
        X = self.gcn(X, edge_index)
        
        # Get the complete graph representation
        X = global_mean_pool(X, batch)

        # Infer the graph class
        X = dropout(X, p=0.3, training=self.training)
        X = self.mlp(X)

        return X 

    def __str__(self):
        return light_blue(f"{self.gcn}\n{self.mlp}")
    
