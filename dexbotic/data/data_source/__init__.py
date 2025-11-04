import os
import importlib
import sys


def import_modules_from_directory(directory, package_prefix=None, is_external=False):
    if not os.path.exists(directory):
        return

    for filename in os.listdir(directory):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]
            if is_external:
                # For external directories, ensure the directory is in sys.path
                if directory not in sys.path:
                    sys.path.insert(0, directory)
                importlib.import_module(module_name)
            else:
                full_module_name = f"{package_prefix}.{module_name}" if package_prefix else module_name
                importlib.import_module(full_module_name)


# Import modules from the current directory
current_dir = os.path.dirname(__file__)
import_modules_from_directory(current_dir, package_prefix=__name__)

# Import modules from external DEXBOTIC_DATA_PATH directory
dexbotic_data_path = os.getenv('DEXBOTIC_DATA_PATH', None)
if dexbotic_data_path:
    import_modules_from_directory(dexbotic_data_path, is_external=True)
