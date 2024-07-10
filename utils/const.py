import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv

GCN_MODEL_SETTINGS = ["gcn_embed_dims", "gcn_layer", "gcn_act", "mlp_embed_dims"] 
TRAINING_SETTINGS  = ["epochs", "learning_rate", "gcn_path"]

ACTIVATION_FUNCTIONS_MAPPING = {"relu": nn.ReLU(), 
                                "leaky_relu": nn.LeakyReLU(), 
                                "sigmoid": nn.Sigmoid(), 
                                "tanh": nn.Tanh()}
CONV_LAYERS_MAPPING          = {"gcn": GCNConv, 
                                "gcnconv": GCNConv, 
                                "sage": SAGEConv, 
                                "sageconv": SAGEConv}