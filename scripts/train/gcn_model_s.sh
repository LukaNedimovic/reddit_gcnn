# Go to parent directory 
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--num_rows 1000 \
--gcn_path "./models/gcn_model_s.pth" \
--gcn_embed_dims 16 8 4 \
--mlp_embed_dims 4 2 1 \
--learning_rate 0.01 \
--epochs 50 \
--test_size 0.2