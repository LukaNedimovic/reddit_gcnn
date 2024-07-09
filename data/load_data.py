import networkx as nx

import torch
from torch_geometric.utils import from_networkx
from torch_geometric       import seed_everything

from torch_geometric.transforms import RandomLinkSplit

def load_data():
    seed_everything(42) # Random seed, for reproduction purposes
    
    full_graph = nx.karate_club_graph()     # Load Zachary's Karate Club dataset, with node attributes  
    graph      = nx.Graph(full_graph.edges) # Create graph with no node attributes
    
    data = from_networkx(graph) # Only store connectivity information

    # Create a train-validation-test split
    split_transform = RandomLinkSplit(num_val=0.1, 
                                      num_test=.2,
                                      is_undirected=True,
                                      add_negative_train_samples=False,
                                      neg_sampling_ratio=1.0)
    train_data, val_data, test_data = split_transform(data)
    
    return (train_data, val_data, test_data)

def load_gcn_model(gcn_model_path: str=None):
    return torch.load(gcn_model_path)