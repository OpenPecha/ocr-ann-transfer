from pathlib import Path 
from typing import List 

def sort_paths_and_get_strings(paths:List[Path]) -> List[str]:
    """ sort paths by string and return as string"""
    return [str(path) for path in sorted(paths, key=lambda path: str(path))]

def sort_paths_and_get_paths(paths:List[Path]) -> List[Path]:
    """ sort paths by string and return as string"""
    return [path for path in sorted(paths, key=lambda path: str(path))]