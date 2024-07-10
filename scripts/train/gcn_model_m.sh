# Go to parent directory 
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "./models/gcn_model_m.pth" \
--gcn_embed_dims 256 128 128 128 64 \
--mlp_embed_dims 64 32 16 16 1 \
--learning_rate 0.01 \
--epochs 50 \
--test_size 0.2