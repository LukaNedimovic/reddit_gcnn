import torch
from torch_geometric       import seed_everything
from torch_geometric.data  import Data

from datasets import load_dataset

def convert_to_graph_data(dataset):
    data = []

    for row in dataset:
        iter_edge_index = row["edge_index"]
        label           = row["y"]
        
        # Convert edges into meaningful node pairs
        edge_pairs = [(edge_u, edge_v) for edge_u, edge_v in zip(iter_edge_index[0], iter_edge_index[1])]
        
        # Create an adequate pairing as torch tensor
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

        # Create a Data object
        graph_data = Data(edge_index=edge_index, y=torch.tensor([label]), max_node=edge_index.max().item() + 1)

        # Add it to the complete dataset
        data.append(graph_data)

    return data

def load_data(num_rows:  int=1_000,
              test_size: float=0.5):
    seed_everything(42) # Random seed, for reproduction purposes
    
    dataset = load_dataset("graphs-datasets/reddit_threads")
    dataset = dataset["full"].select(range(num_rows)).train_test_split(test_size=test_size)

    train_data = convert_to_graph_data(dataset["train"])
    test_data  = convert_to_graph_data(dataset["test"])

    return (train_data, test_data)

def load_gcn_model(gcn_model_path: str=None):
    return torch.load(gcn_model_path)