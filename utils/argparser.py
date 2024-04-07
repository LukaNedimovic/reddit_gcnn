import argparse # Parsing arguments
from .log import * # Colorful output

# Help texts, so the lines don't get too long
HELP = {
    "gnn_layers": "Specify the number of GNN layers in model.",
    "script_name": "Specify the name of the script used to run program.",
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
    
    # GNN layers must be present
    assert args.gnn_layers > 0, red("GNN layers number must be integer greater than 0.")
    
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

    # Model arguments
    parser.add_argument("--gnn_layers", dest="gnn_layers", action="store", type=int, default=1, help=HELP["gnn_layers"])

    # Parse arguments
    args, unknown_args = parser.parse_known_args()
    
    # Assure the user passed arguments properly
    test(args, unknown_args)
    
    print_done("Arguments loaded.")
    
    # In case of script use, log it
    if args.script_name is not None:
        print_warning(f"This program has been run using following script: {args.script_name}")
    
    
    # Returns passed arguments
    return args