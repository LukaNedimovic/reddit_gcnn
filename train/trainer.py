import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from sklearn.metrics import roc_auc_score

import progressbar
from utils.log import * # Colorful output

def eval_gcn_model(gcn_model, features, data):
    gcn_model.eval()
    with torch.no_grad():
        # Generate embeddings, then use them to make predictions for all the validation edges
        node_embeds = gcn_model.encode(features, data.edge_index)
        predictions = gcn_model.decode(node_embeds, data.edge_label_index).view(-1).sigmoid()

        # Return standard ROC-AUC score
        return roc_auc_score(data.edge_label.cpu().numpy(), predictions.cpu().numpy())

def train_gcn_model(gcn_model=None, 
                    data=None,
                    settings: dict=None):

    # Unpack data into respective datasets
    train_data, val_data, test_data = data
    
    # Extract general training settings
    epochs, learning_rate, gcn_path, device = (settings[key] for key in settings)

    # Initialize a random initial embedding
    train_data.x = torch.rand(gcn_model.num_nodes, gcn_model.gcn_embed_dims[0], 
                              requires_grad=True).to(gcn_model.device)

    # Port data to adequate device    
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    # Loss function and optimizer setup
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=learning_rate)
    
    with progressbar.ProgressBar(max_value=epochs, redirect_stdout=True) as bar:
        for epoch in range(epochs):
            gcn_model.train()
            optimizer.zero_grad()
            
            # Retrieve the node embeddings for all the nodes present in the original graph
            node_embeds = gcn_model.encode(train_data.x, train_data.edge_index)
            
            # Generate random negative edges
            neg_edge_index = negative_sampling(edge_index=train_data.edge_index,
                                               num_nodes=train_data.num_nodes,
                                               num_neg_samples=train_data.edge_label_index.size(1), 
                                               method="sparse")
            
            # Add newly generated negative edges into the training dataset
            edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index],
                                         dim=-1)
            # Generate the labels for newly created negative edges - 0, as opposed to 1 for the positive edges
            edge_label = torch.cat([train_data.edge_label,train_data.edge_label.new_zeros(neg_edge_index.size(1))], 
                                   dim=0)

            # Generate predictions for both positive and negative edges, by passing them through MLP
            predictions = gcn_model.decode(node_embeds, edge_label_index).view(-1)
            loss        = criterion(predictions, edge_label)
            
            # Backpropagate
            loss.backward()
            optimizer.step()

            # Calculate Validation ROC-AUC and print it every 10th epoch
            val_roc_auc = eval_gcn_model(gcn_model, train_data.x, val_data)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch:4d}. Train loss: {loss:.4f}. Validation ROC-AUC score: {val_roc_auc:.4f}")

            # Update the progress bar
            bar.update(epoch)

    print_done("Model training has been finished.")

    # Training is finished. Test the model on the test data.
    test_auc = eval_gcn_model(gcn_model, train_data.x, test_data)
    print_info(f"Test AUC: {test_auc:.2f}")

    # Save the model on disk
    print_info(f"Saving the model...")
    torch.save(gcn_model, gcn_path)
    print_info(f"Model successfully saved.")