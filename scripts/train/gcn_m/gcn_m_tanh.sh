# Go to parent directory 
cd ..
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "./models/gcn_m_tanh.pth" \
--gcn_embed_dims 256 128 128 128 64 \
--gcn_layer "GCNConv" \
--gcn_act "tanh" \
--mlp_embed_dims 64 32 16 16 1 \
--learning_rate 0.001 \
--epochs 20 \
--test_size 0.5
