from pathlib import Path


def get_package_root():
    """
    Returns the absolute path to the RL-SCS package root directory.
    
    Returns:
        Path: Absolute path to the src/ directory
    """
    return Path(__file__).parent.absolute()

