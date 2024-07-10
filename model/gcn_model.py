import torch
import torch.nn as nn
from torch.nn.functional import dropout

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool

from utils.log import *
from utils.const import GCN_MODEL_SETTINGS, CONV_LAYERS_MAPPING, ACTIVATION_FUNCTIONS_MAPPING

class GCN(nn.Module):
    def __init__(self, 
                 num_nodes:  int=None,
                 embed_dims: list=[1, 1],
                 layer:      str="GCN",
                 act:        str="ReLU"):
        super(GCN, self).__init__()

        # Create the node embedding layer
        self.emb            = nn.Embedding(num_nodes, embed_dims[0])

        # General graph convolution data
        self.num_gcn_layers = len(embed_dims) - 1
        self.embed_dims     = embed_dims
        self.insert_act     = len(act) != 0
        
        # Extract type of convolution and activation function between layers
        self.layer          = CONV_LAYERS_MAPPING.get(layer.lower(), GCNConv)
        self.act            = ACTIVATION_FUNCTIONS_MAPPING.get(act.lower(), nn.ReLU());

        self.setup_layers()

    def setup_layers(self):
        # Setup the GCN layers
        gcn_layers = []
        for layer_idx in range(self.num_gcn_layers):
            gcn_layers.append(self.layer(self.embed_dims[layer_idx], self.embed_dims[layer_idx + 1]))

            # Append an activation function if the model requires it
            # and the layer is not the last one
            if self.insert_act and layer_idx != self.num_gcn_layers - 1:
                gcn_layers.append(self.act)
        
        # Create a graph convolutional layer
        self.gcn = nn.ModuleList(gcn_layers)
        
    def forward(self, edge_index):
        node_embeds = self.emb.weight

        for layer in self.gcn:
            if self.is_convolution(layer):
                node_embeds = layer(node_embeds, edge_index)
            else:
                node_embeds = layer(node_embeds)

        return node_embeds

    def is_convolution(self, layer):
        return isinstance(layer, GCNConv) or isinstance(layer, SAGEConv)

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
        return self.mlp(X).sigmoid()

class GCNModel(nn.Module):
    def __init__(self, settings):
        super(GCNModel, self).__init__()

        gcn_embed_dims, gcn_layer, gcn_act, mlp_embed_dims = (settings[key] for key in GCN_MODEL_SETTINGS)
        num_nodes = settings["num_nodes"]
        
        # Initialize layers
        self.gcn = GCN(num_nodes, gcn_embed_dims, gcn_layer, gcn_act)
        self.mlp = MLP(mlp_embed_dims)
    
        # Keep the data
        self.gcn_embed_dims = gcn_embed_dims
        self.gcn_layer      = gcn_layer
        self.gcn_act        = gcn_act
        self.mlp_embed_dims = mlp_embed_dims
        
        # Port the model to adequate device (preferably CUDA, otherwise CPU)
        self.port_to_device()

    def port_to_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.device = device
        
        print_done("Model ported to device: " + str(device))

    def forward(self, edge_index, batch):
        # Infer embeddings
        node_embeds   = self.gcn(edge_index)
        unique_embeds = node_embeds[torch.unique(edge_index)]        
        
        # Get the complete graph representation
        unique_embeds = global_mean_pool(unique_embeds, batch)

        # Infer the graph class
        dropout_embeds = dropout(unique_embeds, 
                                 p=0.3, 
                                 training=self.training)

        prediction = self.mlp(dropout_embeds).view(-1)
        return prediction

    def __str__(self):
        return light_blue(f"{self.gcn}\n{self.mlp}")
    
