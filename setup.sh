echo "Starting environment setup."

# Export environment variables (for the sake of scripts)
export PROJECT_DIR="$HOME/Desktop/pmf_exp_nn_1_proj"
export DATA_DIR="$PROJECT_DIR/data"
export MODEL_DIR="$PROJECT_DIR/model"
export MODEL_CACHE_DIR="$PROJECT_DIR/models"
export TRAIN_DIR="$PROJECT_DIR/train"
export SCRIPTS_DIR="$PROJECT_DIR/scripts"
export UTILS_DIR="$PROJECT_DIR/utils"

export PYTHONPATH="$PROJECT_DIR"

# Navigate to project root
cd $PROJECT_DIR

# Load the venv
source venv/bin/activate

# Install requirements for venv
pip install -r requirements.txt

# Make a directory for models to be stored at
mkdir "$PROJECT_DIR/models"

echo "Environment setup finished."