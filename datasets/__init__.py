
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

# Import every file under the folder for the registry to work

from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module

IGNORED_MODULES = {'base_dataset', 'transforms'}

# iterate recursively through the modules in the current package
def iter_submodules(path, prefix=''):
    for _, name, is_pkg in iter_modules([path]):
        if name in IGNORED_MODULES:
            continue
        full_name = prefix + '.' + name
        yield full_name, is_pkg
        if is_pkg:
            yield from iter_submodules(Path(path)/name, full_name)

# import all modules in the current package
root = Path(__file__).parent
all_modules = iter_submodules(root, "datasets")
for module_name, is_pkg in all_modules:
    if is_pkg:
        continue
    module = import_module(module_name)
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)

        if isclass(attribute):            
            # Add the class to this package's variables
            globals()[attribute_name] = attribute

