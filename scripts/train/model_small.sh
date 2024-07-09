# CONSTANTS
SCRIPT_NAME="model_small.sh"

# Go to parent directory 
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$SCRIPT_NAME" \
--gcn_embed_dims 3 3 1 \
--mlp_embed_dims 2 1 \
--learning_rate 0.001 \
--epochs 500