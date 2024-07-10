# Go to parent directory 
cd ..
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "./models/gcn_l_relu.pth" \
--gcn_embed_dims 2048 2048 2048 2048 1024 1024 512 \
--gcn_layer "GCNConv" \
--gcn_act "relu" \
--mlp_embed_dims 512 256 128 1 \
--learning_rate 0.0001 \
--epochs 20 \
--test_size 0.5