import os
from pathlib import Path


def get_or_create_dir(path):
    """ Creates directories associated to the specified path if missing, and returns the Path object. """
    path = Path(path)
    if not path.exists():
        os.makedirs(path)
    return path


DATA_DIR = get_or_create_dir("DATA")
GO_PATH = DATA_DIR / "go-basic.obo"

NAMESPACES = ("biological_process", "molecular_function", "cellular_component")
