from pathlib import Path
import importlib.resources


def get_package_root():
    """
    Returns the absolute path to the RL-SCS package root directory (the src/ folder).
    """
    try:
        with importlib.resources.path('src', '__init__.py') as p:
            return p.parent
    except (ImportError, FileNotFoundError):
        # If the package is not installed we just use the relative path
        return Path(__file__).resolve().parent.parent

