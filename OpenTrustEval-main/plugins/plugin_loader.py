import importlib
import os
import pkgutil

def load_plugins(plugin_dir='plugins'):
    """
    Dynamically load all plugin modules from the given directory.
    Returns:
        dict: {plugin_name: module}
    """
    plugins = {}
    for _, name, ispkg in pkgutil.iter_modules([plugin_dir]):
        if not ispkg and not name.startswith('__'):
            module = importlib.import_module(f'{plugin_dir}.{name}'.replace('/', '.'))
            plugins[name] = module
    return plugins
