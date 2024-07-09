import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from utils.log import *
from utils.const import GCN_MODEL_SETTINGS

class GCN(nn.Module):
    def __init__(self, 
                 gcn_layers: int=1, 
                 embed_dims: list=[1, 1]):
        super(GCN, self).__init__()

        assert embed_dims is not None and len(embed_dims) == gcn_layers + 1, red("Number of embedding sizes must be equal to GCN_LAYERS + 1.")

        self.num_gcn_layers = gcn_layers
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
                 fc_layers: int=1, 
                 dims:      list=[1, 1]):
        
        super(MLP, self).__init__()

        assert dims is not None and len(dims) == fc_layers + 1, red("Number of embedding sizes must be equal to GCN_LAYERS + 1.")

        self.num_fc_layers = fc_layers
        self.dims          = dims
        
        self.setup_layers()

    def setup_layers(self):
        # Setup the GCN layers
        mlp_layers = []
        for layer_idx in range(self.num_fc_layers):
            mlp_layers.append(nn.Linear(self.dims[layer_idx], self.dims[layer_idx + 1]))
            
            if layer_idx != self.num_fc_layers - 1:
                mlp_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, X):
        X = self.mlp(X)

        return X

class GCNModel(nn.Module):
    def __init__(self, settings):
        super(GCNModel, self).__init__()

        gcn_embed_dims, mlp_embed_dims = (settings[key] for key in GCN_MODEL_SETTINGS)
        num_nodes = settings["num_nodes"]
        
        gcn_layers = len(gcn_embed_dims) - 1
        mlp_layers = len(mlp_embed_dims) - 1
        
        # Initialize layers
        self.gcn = GCN(gcn_layers, gcn_embed_dims)
        self.mlp = MLP(mlp_layers, mlp_embed_dims)
    
        # Keep the data
        self.num_nodes      = num_nodes
        self.gcn_embed_dims = gcn_embed_dims
        self.mlp_embed_dims = mlp_embed_dims
        
        # Port the model to adequate device (preferably CUDA, otherwise CPU)
        self.port_to_device()

    def port_to_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.device = device
        
        print_done("Model ported to device: " + str(device))

    def encode(self, X, edges):
        X = self.gcn(X, edges)

        return X 

    def decode(self, X, edges):
        return self.mlp(torch.cat([X[edges[0]], X[edges[1]]], dim=-1))
    
    def __str__(self):
        return light_blue(f"{self.gcn}\n{self.mlp}")
    
