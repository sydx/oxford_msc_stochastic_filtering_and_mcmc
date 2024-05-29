def checknotnone(arg, message="'None' not allowed"):
    if arg is None: raise ValueError(message)
    return arg

def checknonnegativeinteger(arg, message='Not a nonnegative integer'):
    if not isinstance(arg, int): raise ValueError(message)
    return arg
    