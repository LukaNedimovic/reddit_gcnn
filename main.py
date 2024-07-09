from utils.argparser import parse_args
from utils.const import *

from model.gcn_model import GCNModel

from train.trainer import *
from data.load_data import load_data, load_gcn_model

if __name__ == "__main__":
    # Parse arguments from command line
    args = parse_args()
    
    # Create classical Graph Convolutional Network model
    gcn_model  = None
    qgcn_model = None

    if args.train_gcn:
        # Load Zachary's Karate Club Dataset
        data = load_data()
        
        gcn_model_settings              = {key: vars(args)[key] for key in GCN_MODEL_SETTINGS}
        gcn_model_settings["num_nodes"] = data[0].num_nodes
        
        gcn_model = GCNModel(settings=gcn_model_settings)
        
        training_settings           = {key: vars(args)[key] for key in TRAINING_SETTINGS}
        training_settings["device"] = gcn_model.device
        
        train_gcn_model(gcn_model=gcn_model,
                        data=data,
                        settings=training_settings)

    elif args.load_gcn:
        print_done(f"Staring model load: {args.gcn_path}")        
        gcn_model = load_gcn_model(args.gcn_path)
        print_done(f"Successfully loaded model: {args.gcn_path}")