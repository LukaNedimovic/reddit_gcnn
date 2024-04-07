from termcolor import colored

def red(text: str) -> str:
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
    Returns cyan text.
    
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