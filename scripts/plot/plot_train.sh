#!/bin/bash

"$UTILS_DIR/plot.py" --csv_path "$DATA_DIR/training_data.csv" \
                     --prefix   "train" \
                     --data_id  "train_accuracies" \
                     --x_label  "Epoch" \
                     --y_label  "Training Accuracy" \