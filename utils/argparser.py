import argparse    # Parsing arguments
from .log import * # Colorful output

# Help texts, so the lines don't get too long
HELP = {
    "script_name":    "Specify the name of the script used to run program.",
    "train_gcn":      "Specify to train the GCN model.",
    "load_gcn":       "Specify to load the GCN model.",
    "gcn_path":       "Specify the GCN model path - be it for loading from, or saving.",
    "epochs":         "Specify the number of epochs for the model to be trained.",
    "learning_rate":  "Specify the learning rate used for training.",
    "gcn_embed_dims": "Specify the embedding dimensions in each layer of GCNConv.",
    "mlp_embed_dims": "Specify the embedding dimensions in each layer of MLP."
}


def test(args: argparse.Namespace, unknown_args: argparse.Namespace):
    """
    Test passed cmdline arguments using assertions.
    If everything is alright, program continues as planned. 
    If not, program stops upon assertion error.
    
    Parameters
    ----------
    args : argparse.Namespace
        Known parsed arguments.
        
    unknown_args: argparse.Namespace
        Unknown parsed arguments.
    
    """
    
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

        # First MLP layer handles 2 node representations, therefore its dimension is twice as big compared to GCN output's dimension
        assert 2 * args.gcn_embed_dims[-1] == args.mlp_embed_dims[0], red(f"GCN output ({args.gcn_embed_dims[-1]}) and MLP input ({args.mlp_embed_dims[0]}) shape mismatch.")

    print_done("Arguments testing finished.")


def parse_args() -> argparse.Namespace:
    """
    Parses cmdline arguments and returns them to the main function.
    
    Returns
    -------
    args : argparse.Namespace
        Parsed arguments.
        Program will not procceed if any unknown argument is parsed.
    """
    parser = argparse.ArgumentParser(prog="AI Project CHANGE THIS TODO: CHANGE",
                                     description="Parses your arguments TODO: CHANGE",
                                     epilog="Help TODO: CHANGE")

    # General
    parser.add_argument("--script_name", dest="script_name", action="store", type=str, default=None, help=HELP["script_name"])

    # Training environment arguments
    parser.add_argument("--train_gcn",      dest="train_gcn",     action="store_true",                            help=HELP["train_gcn"])
    parser.add_argument("--load_gcn",       dest="load_gcn",      action="store_true",                            help=HELP["load_gcn"])
    parser.add_argument("--gcn_path",       dest="gcn_path",      action="store",      type=str,   default=None,  help=HELP["gcn_path"])
    parser.add_argument("--epochs",         dest="epochs",        action="store",      type=int,   default=5,     help=HELP["epochs"])
    parser.add_argument("--learning_rate",  dest="learning_rate", action="store",      type=float, default=0.001, help=HELP["learning_rate"])

    # Model arguments
    parser.add_argument("--gcn_embed_dims", nargs="*", type=int, default=[1, 1], help=HELP["gcn_embed_dims"])
    parser.add_argument("--mlp_embed_dims", nargs="*", type=int, default=[1, 1], help=HELP["mlp_embed_dims"])

    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    
    # Assure the user passed arguments properly
    test(args, unknown_args)
    
    print_done("Arguments loaded.")
    
    # In case of script use, log it
    if args.script_name is not None:
        print_info(f"This program has been run using following script: {args.script_name}")
    
    # Returns passed arguments
    return args