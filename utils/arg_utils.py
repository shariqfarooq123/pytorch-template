
# MIT License

# Copyright (c) 2022 Shariq Farooq Bhat

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



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
