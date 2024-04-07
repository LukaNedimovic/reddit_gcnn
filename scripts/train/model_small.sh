# CONSTANTS
SCRIPT_NAME="model_small.sh"
GNN_LAYERS=1

# Go to parent directory 
cd ..
cd ..

# Execute the script
python main.py --script_name "$SCRIPT_NAME" --gnn_layers "$GNN_LAYERS"
