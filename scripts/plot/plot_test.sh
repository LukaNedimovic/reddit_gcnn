#!/bin/bash

"$UTILS_DIR/plot.py" --csv_path "$DATA_DIR/training_data.csv" \
                     --prefix   "test" \
                     --data_id  "test_accuracies" \
                     --x_label  "Epoch" \
                     --y_label  "Test Accuracy" \