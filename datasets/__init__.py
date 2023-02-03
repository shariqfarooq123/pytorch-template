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

