import numpy as np 
from pathlib import Path 
from PIL import Image
from typing import List 
from tqdm import tqdm 

from ocr_ann_transfer.monlam_ocr.line_detection import LineDetection
from ocr_ann_transfer.utility import get_image_as_array
from ocr_ann_transfer.config import MODEL_CONFIG_PATH

def load_model(model_config_path:Path = MODEL_CONFIG_PATH)->LineDetection:
    return LineDetection(model_config_path)

def run_inference(model:LineDetection, image_array:np.array):
    res = model.predict(image_array)
    return res 

def detect_lines_from_image(image_path:Path, model:LineDetection, destination_dir:Path):
    """ convert image to image array (numpy)"""
    image_array = get_image_as_array(image_path)

    res = run_inference(model, image_array)
    line_images = res[1]

    image_name = image_path.stem 
    image_dir = destination_dir / image_name

    destination_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    for i, line_image in enumerate(line_images, start=1):
        image = Image.fromarray(line_image)
        image.save(image_dir / f"{image_name}_{i:04}.jpg")

def detect_lines_from_images(images_path:List[Path], destination_dir:Path):
    model = load_model()
    for image_path in  tqdm(images_path, desc="detecting lines in images") :
        detect_lines_from_image(image_path, model, destination_dir)

if __name__ == "__main__":
    destination_dir = Path("ocr_output")
    images_path = list(Path("/home/tenzin3/ocr-ann-transfer/images_dir/W2PD17382-I1KG81275").rglob("*.jpg"))
    detect_lines_from_images(images_path, destination_dir)
