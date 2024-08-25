import argparse    # Parsing arguments
from utils.log import * # Colorful output

# Help texts, so the lines don't get too long
HELP = {
    # Parser - STD
    "script_name":    "Specify the name of the script used to run program.",
    "train_gcn":      "Specify to train the GCN model.",
    "load_gcn":       "Specify to load the GCN model.",
    "num_rows":       "Specify to select first N rows from the dataset.",
    "gcn_path":       "Specify the GCN model path - be it for loading from, or saving.",
    "epochs":         "Specify the number of epochs for the model to be trained.",
    "test_size":      "Specify the percentage of samples used in test dataset (e.g. 0.2).",
    "learning_rate":  "Specify the learning rate used for training.",
    "gcn_embed_dims": "Specify the embedding dimensions in each layer of GCNConv.",
    "gcn_layer":      "Specify the convolution layer implemented within GCN.",
    "gcn_act":        "Specify the activation function after each (except the last) layer in GCN.",
    "mlp_embed_dims": "Specify the embedding dimensions in each layer of MLP.",
    # Parser - Plot
    "csv_path":       "Path to csv file for plotting purposes.",
    "data_id":        "Column where tuples of form (x, y) are stored.",
    "x_label":        "Label of the x-axis.",
    "y_label":        "Label of the y-axis.",
    "prefix":         "Prefix of the plot's filename."
}

def test_std(args: argparse.Namespace, unknown_args: argparse.Namespace):
    """ Test standard parser arguments """
    
    print_info("Starting argument testing.")
    
    # No unknown arguments should be passed
    assert len(unknown_args) == 0, red("Unkown cmdline arguments passed.")
    
    # Program must either train or load an already trained GCN model - can't do both
    assert not (args.train_gcn and args.load_gcn), red("Can't both train and load a GCN model.")
    
    # If loading a model
    if args.load_gcn:
        # Loading destination must be provided
        assert args.gcn_path is not None, red("GCN Model path must be provided.")
    
    # If training a model
    elif args.train_gcn:
        # Storage destination must be provided
        assert args.gcn_path is not None, red("GCN Model path must be provided.")
    
        # GCN layers must be present and embedding dimensions must be natural numbers
        assert len(args.gcn_embed_dims) > 0, red("GCN layers number must be integer greater than 0.")
        for embed_dim in args.gcn_embed_dims:
            assert embed_dim > 0, red("GCN layer embedding dimension must be greater than 0.")

        # MLP layers must be present and embedding dimensions must be natural numbers
        assert len(args.mlp_embed_dims) > 0, red("MLP layers number must be integer greater than 0.")
        for embed_dim in args.mlp_embed_dims:
            assert embed_dim > 0, red("MLP layer embedding dimension must be greater than 0.")
        
        assert args.gcn_embed_dims[-1] == args.mlp_embed_dims[0], red("GCN layer and MLP layer dimension mismatch.")  

    print_done("Arguments testing finished.")


def parse_plot():
    """ Setup argparser for plotting purposes """
    
    parser = argparse.ArgumentParser(prog="Plot Parser",
                                     description="Parses your arguments!")
    
    # Plotting arguments
    parser.add_argument("--csv_path", dest="csv_path", action="store", type=str, default=None, help=HELP["csv_path"])
    parser.add_argument("--data_id",  dest="data_id",  action="store", type=str, default=None, help=HELP["data_id"])
    parser.add_argument("--x_label",  dest="x_label",  action="store", type=str, default=None, help=HELP["x_label"])
    parser.add_argument("--y_label",  dest="y_label",  action="store", type=str, default=None, help=HELP["y_label"])
    parser.add_argument("--prefix",   dest="prefix",   action="store", type=str, default=None, help=HELP["prefix"])

    return parser


def parse_std():
    """ Setup a standard argparser (i.e. for training / inference purposes) """

    parser = argparse.ArgumentParser(prog="Standard Parser",
                                     description="Parses your arguments!")

    # General
    parser.add_argument("--script_name", dest="script_name", action="store", type=str, default=None, help=HELP["script_name"])

    # Training environment arguments
    parser.add_argument("--train_gcn",      dest="train_gcn",     action="store_true",                            help=HELP["train_gcn"])
    parser.add_argument("--load_gcn",       dest="load_gcn",      action="store_true",                            help=HELP["load_gcn"])
    parser.add_argument("--num_rows",       dest="num_rows",      action="store",      type=int,   default=1_000, help=HELP["num_rows"])
    parser.add_argument("--gcn_path",       dest="gcn_path",      action="store",      type=str,   default=None,  help=HELP["gcn_path"])
    parser.add_argument("--epochs",         dest="epochs",        action="store",      type=int,   default=5,     help=HELP["epochs"])
    parser.add_argument("--learning_rate",  dest="learning_rate", action="store",      type=float, default=0.01, help=HELP["learning_rate"])
    parser.add_argument("--test_size",      dest="test_size",     action="store",      type=float, default=0.5,   help=HELP["test_size"])

    # Model arguments
    parser.add_argument("--gcn_embed_dims", nargs="*",        type=int, default=[1, 1],    help=HELP["gcn_embed_dims"])
    parser.add_argument("--gcn_layer",      dest="gcn_layer", type=str, default="GCNConv", help=HELP["gcn_layer"])
    parser.add_argument("--gcn_act",        dest="gcn_act",   type=str, default="ReLU",    help=HELP["gcn_act"])
    
    parser.add_argument("--mlp_embed_dims", nargs="*",        type=int, default=[1, 1],      help=HELP["mlp_embed_dims"])

    return parser


PARSER_MAPPING = {
    "std":  (parse_std,  test_std),
    "plot": (parse_plot, None)
}

def parse_args(parser_id: str="std") -> argparse.Namespace:
    """ Parses cmdline arguments and returns them. """
    
    assert parser_id in PARSER_MAPPING, red(f"Invalid parser_id ({parser_id}) provided.")
    setup, test = PARSER_MAPPING[parser_id]

    # Set up the adequate parser
    parser = setup()

    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    
    # Assure the user passed arguments properly
    if test is not None:
        test(args, unknown_args)
    
    print_done("Arguments loaded.")
    
    # In case of script use, log it
    if "script_name" in vars(args) and args.script_name is not None:
        print_info(f"This program has been run using following script: {args.script_name}")
    
    # Returns passed arguments
    return args