from termcolor import colored

    
def red(text: str=None) -> str:
    """
    Returns red text.
    
    Parameters
    ----------
    text : str
        Text to be turned red.
        
    Returns
    -------
    _ : str
        Red text.
    """
    assert isinstance(text, str), colored("Text must be a string.")
    
    return colored(text, "red")


def green(text: str) -> str:
    """
    Returns green text.
    
    Parameters
    ----------
    text : str
        Text to be turned green.
        
    Returns
    -------
    _ : str
        green text.
    """
    assert isinstance(text, str), colored("Text must be a string.")
    
    return colored(text, "green")


def blue(text: str) -> str:
    """
    Returns blue text.
    
    Parameters
    ----------
    text : str
        Text to be turned blue.
        
    Returns
    -------
    _ : str
        blue text.
    """
    assert isinstance(text, str), colored("Text must be a string.")
    
    return colored(text, "blue")


def light_blue(text: str) -> str:
    """
    Returns light_blue text.
    
    Parameters
    ----------
    text : str
        Text to be turned light_blue.
        
    Returns
    -------
    _ : str
        light_blue text.
    """
    assert isinstance(text, str), colored("Text must be a string.")
    
    return colored(text, "light_blue")


def yellow(text: str) -> str:
    """
    Returns yellow text.
    
    Parameters
    ----------
    text : str
        Text to be turned yellow.
        
    Returns
    -------
    _ : str
        yellow text.
    """
    assert isinstance(text, str), colored("Text must be a string.")
    
    return colored(text, "yellow")


COLOR_FUNCTIONS = {"red": red,
                   "green": green,
                   "blue": blue,
                   "light_blue": light_blue}


def print_done(text: str=None):
    """
    Prints green text with a checkmark in front, to symbolize that certain process is done.
    
    Parameters
    ----------
    text : str
        Text to be printed.
    """
    
    assert isinstance(text, str), red("[PRINT_CHECK] Text must be a string.")

    print(green(f"[✅] {text}"))
  
  
def print_info(text: str=None):
    """
    Prints blue text with an info symbol in front, to show some information.
    
    Parameters
    ----------
    text : str
        Text to be printed.
    """
    
    assert isinstance(text, str), red("[PRINT_INFO] Text must be a string.")

    print(light_blue(f"[ℹ️] {text}"))
    
def print_warning(text: str=None):
    """
    Prints yellow text with an info symbol in front, to symbolize that certain process is done.
    
    Parameters
    ----------
    text : str
        Text to be printed.
    """
    
    assert isinstance(text, str), red("[PRINT_WARNING] Text must be a string.")

    print(yellow(f"[⚠️] {text}"))
      

def log(header: str=None, text: str=None, color: str=None):
    """
    Logs text in format "[HEADER]: Text" (in given color).
    
    Parameters
    ----------
    header : str (None)
        Header to be printed.
    text : str (None)  
        Text to be printed.
    color : str (None)
        Color of text to be printed.
    
    """
    
    assert isinstance(header, str), red("[LOG]: Header must be a string.")
    assert isinstance(text, str), red("[LOG]: Text must be a string.")
    assert isinstance(color, str), red("[LOG]: Color must be a string.")
    assert (color in ["red", "green", "blue", "light_blue"]), red("[LOG]: Color must be a valid one.")
    
    color = COLOR_FUNCTIONS[color]
    
    print(color(f"[{header}]: {text}"))