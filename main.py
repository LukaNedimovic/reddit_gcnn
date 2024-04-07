from utils.argparser import parse_args

from model.model import Model

if __name__ == "__main__":
    args = parse_args()
    
    model = Model(gnn_layers=args.gnn_layers)
    
    print(model)
    