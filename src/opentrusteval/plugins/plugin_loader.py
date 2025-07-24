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


def load_high_performance_plugins():
    """
    Load plugins specifically designed for the high-performance system.
    Returns:
        dict: {plugin_name: module}
    """
    high_perf_plugins = {}
    
    # Load from high_performance_system/plugins if it exists
    high_perf_plugin_dir = '../high_performance_system/plugins'
    if os.path.exists(high_perf_plugin_dir):
        for _, name, ispkg in pkgutil.iter_modules([high_perf_plugin_dir]):
            if not ispkg and not name.startswith('__'):
                try:
                    module = importlib.import_module(f'high_performance_system.plugins.{name}')
                    high_perf_plugins[f'high_perf_{name}'] = module
                except Exception as e:
                    print(f"Warning: Failed to load high-performance plugin {name}: {e}")
    
    return high_perf_plugins

def get_plugin_compatibility_info():
    """
    Get information about plugin compatibility with high-performance system.
    Returns:
        dict: Compatibility information
    """
    compatibility_info = {
        'legacy_plugins': [],
        'high_performance_plugins': [],
        'compatible_plugins': []
    }
    
    # Check legacy plugins
    legacy_plugins = load_plugins()
    for name, module in legacy_plugins.items():
        if hasattr(module, 'is_high_performance_compatible'):
            if module.is_high_performance_compatible():
                compatibility_info['compatible_plugins'].append(name)
            else:
                compatibility_info['legacy_plugins'].append(name)
        else:
            compatibility_info['legacy_plugins'].append(name)
    
    # Check high-performance plugins
    high_perf_plugins = load_high_performance_plugins()
    compatibility_info['high_performance_plugins'] = list(high_perf_plugins.keys())
    
    return compatibility_info
