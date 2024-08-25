import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import os

import progressbar
from utils.log import * # Colorful output

def evaluate_model(gcn_model, loader):
    gcn_model.eval()

    matches = 0
    for data in loader: # Iterate in batches over the training/test dataset
        # Make predictions and turn them into 0 - 1 labels, by taking the label with higher probability
        predictions = gcn_model(data.edge_index, data.batch)  
        predictions = predictions.round().int()

        # Check predictions against ground-truth labels
        matches += int((predictions == data.y).sum())

    return (matches / len(loader.dataset))

def train_gcn_model(gcn_model=None, 
                    data=None,
                    settings: dict=None):

    # Unpack data into respective datasets
    train_data, test_data, num_nodes = data
    
    # Extract general training settings
    epochs, learning_rate, gcn_path, device = (settings[key] for key in settings)
    
    for graph_data in train_data:
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.y          = graph_data.y.to(device)

    for graph_data in test_data:
        graph_data.edge_index = graph_data.edge_index.to(device)
        graph_data.y          = graph_data.y.to(device)

    # Creat data loaders with batches
    train_loader = DataLoader(train_data, batch_size=1, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=1, shuffle=False)

    # Loss function and optimizer setup
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(gcn_model.parameters(), 
                                 lr=learning_rate)

    train_accs = []
    test_accs  = []

    epoch_bar = progressbar.ProgressBar(max_value=epochs, widgets=[progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()], redirect_stdout=True)
    for epoch in range(epochs):
        gcn_model.train()
        
        for data in train_loader:    
            optimizer.zero_grad()
            
            # Infer the prediction by embedding the nodes and passing through MLP
            prediction = gcn_model(data.edge_index, data.batch)
            
            # Calculate the loss
            loss = criterion(prediction, data.y.float())
            
            # Propagate backwards
            loss.backward()
            optimizer.step()

        # Evaluate model on training and testing data after each epoch
        train_acc = evaluate_model(gcn_model, train_loader)
        test_acc  = evaluate_model(gcn_model, test_loader)
        
        # Store the data for future plotting
        train_accs.append((epoch, train_acc))
        test_accs.append((epoch, test_acc))

        # Print epoch summary
        print_info(f"Epoch: {epoch + 1}/{epochs}. Train accuracy: {train_acc * 100: 0.2f}%. Test accuracy: {test_acc * 100: 0.2f}%")
        
        # Update outer progress bar
        epoch_bar.update(epoch + 1)

    # Finish outer progress bar
    epoch_bar.finish()

    print_done("Model training has been finished. ")
    print_info(f"Trained for {epochs} epochs. Train accuracy: {train_accs[-1][1] * 100: 0.2f}%. Test accuracy: {test_accs[-1][1] * 100: 0.2f}%")

    gcn_model_name = os.path.splitext(os.path.basename(gcn_path))[0]

    training_data = [gcn_model_name, 
                     gcn_model.gcn_embed_dims, 
                     gcn_model.gcn_layer,
                     gcn_model.gcn_act,
                     gcn_model.mlp_embed_dims,
                     learning_rate,
                     epochs,
                     train_accs,
                     test_accs]
    
    training_data_path = os.path.expandvars("$DATA_DIR/training_data.csv")
    write_model_training_data(training_data_path, training_data)

    # Save the model on disk
    print_info(f"Saving the model...")
    torch.save(gcn_model, gcn_path)
    print_info(f"Model successfully saved.")