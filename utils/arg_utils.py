

def infer_type(x):  # hacky way to infer type from string args
    """Infers data type of argument from string

    Order of casting : int, float

    Args:
        x (any): input value

    Returns:
        any: casted value
    """
    if not isinstance(x, str):
        return x

    try:
        x = int(x)
        return x
    except ValueError:
        pass

    try:
        x = float(x)
        return x
    except ValueError:
        pass

    return x


def parse_unknown(unknown_args):
    """Parses unknown arguments from argparse

    Args:
        unknown_args (List[str]): Unknown arguments from argparse (e.g. sys.argv[1:])

    Returns:
        dict: Parsed arguments as key-value pairs
    """
    clean = []
    for a in unknown_args:
        if "=" in a:
            k, v = a.split("=")
            clean.extend([k, v])
        else:
            clean.append(a)

    keys = clean[::2]
    values = clean[1::2]
    return {k.replace("--", ""): infer_type(v) for k, v in zip(keys, values)}
