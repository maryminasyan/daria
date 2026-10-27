from .spherex import SPHEREx

def get_instrument(arg):
    if isinstance(arg,str):
        return SPHEREx(field=arg)
    else:
        return arg
