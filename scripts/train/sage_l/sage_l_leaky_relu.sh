#!/bin/bash

# Go to parent directory 
cd $PROJECT_DIR

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "$MODEL_CACHE_DIR/sage_l_leaky_relu.pth" \
--gcn_embed_dims 2048 2048 2048 2048 1024 1024 512 \
--gcn_layer "SAGEConv" \
--gcn_act "leaky_relu" \
--mlp_embed_dims 512 256 128 1 \
--learning_rate 0.0001 \
--epochs 20 \
--test_size 0.5
