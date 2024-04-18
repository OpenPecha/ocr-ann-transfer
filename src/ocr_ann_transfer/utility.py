import numpy as np 
from typing import List 
from pathlib import Path 
from PIL import Image

def sort_paths_and_get_strings(paths:List[Path]) -> List[str]:
    """ sort paths by string and return as string"""
    return [str(path) for path in sorted(paths, key=lambda path: str(path))]

def sort_paths_and_get_paths(paths:List[Path]) -> List[Path]:
    """ sort paths by string and return as string"""
    return [path for path in sorted(paths, key=lambda path: str(path))]

def get_image_as_array(image_path:Path) -> np.array:
    """ convert image to image array (numpy)"""
    image = Image.open(image_path)
    image_array = np.array(image)
    
    return image_array