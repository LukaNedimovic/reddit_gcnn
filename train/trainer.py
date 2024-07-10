import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from itertools import chain

import progressbar
from utils.log import * # Colorful output

def evaluate_model(gcn_model, embeds, loader):
    gcn_model.eval()

    matches = 0
    for data in loader: # Iterate in batches over the training/test dataset
        # Make predictions and turn them into 0 - 1 labels, by taking the label with higher probability
        predictions = gcn_model(embeds, data.edge_index, data.batch)  
        predictions = predictions.round().int()

        # Check predictions against ground-truth labels
        matches += int((predictions == data.y).sum())

    return (matches / len(loader.dataset))

def train_gcn_model(gcn_model=None, 
                    data=None,
                    settings: dict=None):

    # Unpack data into respective datasets
    train_data, test_data, max_node = data
    
    # Extract general training settings
    epochs, learning_rate, gcn_path, device = (settings[key] for key in settings)
    
    # Initialize a random initial embedding 
    X = torch.rand((max_node, gcn_model.gcn_embed_dims[0]), requires_grad=True, device=device)
    
    for graph_data in train_data:
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.y = graph_data.y.to(device)

    for graph_data in test_data:
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.y = graph_data.y.to(device)


    # Creat data loaders with batches
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=1, shuffle=False)

    # Loss function and optimizer setup
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(
                                [
                                    {'params': gcn_model.parameters()},
                                    {'params': X}
                                ], lr=learning_rate
                                )

    epoch_bar = progressbar.ProgressBar(max_value=epochs, widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()], redirect_stdout=True)
    for epoch in range(epochs):
        gcn_model.train()
        
        for data in train_loader:    
            optimizer.zero_grad()
            
            # Infer the prediction by embedding the nodes and passing through MLP
            prediction = gcn_model(X, data.edge_index, data.batch)
            
            # Calculate the loss
            loss = criterion(prediction, data.y.float())
            
            # Propagate backwards
            loss.backward()
            optimizer.step()

        # Evaluate model on training and testing data after each epoch
        train_acc = evaluate_model(gcn_model, X, train_loader)
        test_acc  = evaluate_model(gcn_model, X, test_loader)
        
        # Print epoch summary
        print_info(f"Epoch: {epoch + 1}/{epochs}. Train accuracy: {train_acc * 100: 0.4f}%. Test accuracy: {test_acc * 100: 0.4f}%")
        
        # Update outer progress bar
        epoch_bar.update(epoch + 1)

    # Finish outer progress bar
    epoch_bar.finish()
            
    print_done("Model training has been finished.")

    # Save the model on disk
    print_info(f"Saving the model...")
    torch.save(gcn_model, gcn_path)
    print_info(f"Model successfully saved.")