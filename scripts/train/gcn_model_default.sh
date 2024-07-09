# Go to parent directory 
cd ..
cd ..

# Execute the script
python3 main.py \
--script_name "$0" \
--train_gcn \
--gcn_path "./models/model_small_test.pth" \
--gcn_embed_dims 3 3 1 \
--mlp_embed_dims 2 1 \
--learning_rate 0.001 \
--epochs 500