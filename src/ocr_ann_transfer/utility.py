import numpy as np 
import csv 
import pandas as pd
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


def get_image_url(image_name, batch_id):
    """ returns amazon s3 buckets link of the uploaded image"""
    image_url = f"https://s3.amazonaws.com/monlam.ai.ocr/line_to_text/{batch_id}/{image_name}"
    return image_url

def add_row_to_csv(row, csv_path):
    if Path(csv_path).exists():
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
    else:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


def standardize_line_texts_to_images_csv_mapping(csv_file_path:Path, batch_id:str, group_id:int, state:str="transcribing"):
    """ parse the line texts to images mapping """
    df = pd.read_csv(csv_file_path)
    images_path = list(df["Image Paths"])
    line_texts_path = list(df["Line Text Paths"])

    """ standardize """
    output_file_path = f"{batch_id}.csv"
    headers = ["id","group_id","batch_id","state","inference_transcript","url"]
    add_row_to_csv(headers, output_file_path)
    for image_path, line_texts_path in zip(images_path, line_texts_path):
        id = Path(image_path).name
        inference_transcript = Path(line_texts_path).read_text(encoding="utf-8")
        image_url = get_image_url(id, f"batch_id")
        row = [id, group_id, batch_id, state, inference_transcript, image_url]
        add_row_to_csv(row, output_file_path)
        row = []