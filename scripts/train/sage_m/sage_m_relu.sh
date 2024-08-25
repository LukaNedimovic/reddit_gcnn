#!/bin/bash

# Go to parent directory 
cd $PROJECT_DIR

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "$MODEL_CACHE_DIR/sage_m_relu.pth" \
--gcn_embed_dims 256 128 128 128 64 \
--gcn_layer "SAGEConv" \
--gcn_act "relu" \
--mlp_embed_dims 64 32 16 16 1 \
--learning_rate 0.001 \
--epochs 20 \
--test_size 0.5
