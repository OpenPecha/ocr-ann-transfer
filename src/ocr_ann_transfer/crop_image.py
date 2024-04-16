from pathlib import Path 
from typing import List, Tuple
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm 

from ocr_ann_transfer.utility import sort_paths_and_get_paths

def parse_page_xml(xml_path:Path)->List[Tuple]:
    """
    Parses a PAGE XML file to extract text line coordinates.
    
    :param xml_path: Path to the PAGE XML file.
    :return: A list of tuples, each representing a set of (left, top, right, bottom) coordinates.
    """
    namespace = {'page': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    coords_list = []
    for textLine in root.findall('.//page:TextLine', namespace):
        points = textLine.find('./page:Coords', namespace).attrib['points']
        points = [tuple(map(int, point.split(','))) for point in points.split()]
        xs, ys = zip(*points)
        coords_list.append((min(xs), min(ys), max(xs), max(ys)))
    
    return coords_list

def crop_image(image_path:Path, crop_coords:Tuple[int,int,int,int]):
    try:
        with Image.open(image_path) as img:
            cropped_img = img.crop(crop_coords)
            return cropped_img
    except Exception as e:
        print(f"Error cropping image {image_path}: {e}")
        return None

def crop_multiple_images(image_path:Path, coords_list:List[Tuple], output_dir:Path):
    images_folder = output_dir/ image_path.stem
    images_folder.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(coords_list, start=1):
        cropped_img = crop_image(image_path, coords)
        if cropped_img:
            cropped_img_path = images_folder / f"{image_path.name.rsplit('.', 1)[0]}_cropped_{i}.jpg"
            cropped_img.save(cropped_img_path)
        else:
            print("Skipping saving due to cropping error.")

def images_cropping_pipeline(images_dir:Path, xml_dir:Path, output_dir:Path):
    """ xml file contains the line images coordinates """
    images = list(images_dir.rglob("*.jpg"))
    xmls = list(xml_dir.rglob("*.xml"))

    if len(images) != len(xmls):
        print("[ERROR]: Number of images and xml files are not equal!")
        return 
    
    images = sort_paths_and_get_paths(images)
    xmls = sort_paths_and_get_paths(xmls)

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path, xml_path in tqdm(zip(images, xmls), total=len(images), desc="Cropping images"):
        coords_list = parse_page_xml(xml_path)
        crop_multiple_images(image_path,coords_list, output_dir)
    
    print(f"[SUCCESS]: images in {str(images_dir)} cropped successfully.")
    




if __name__ == "__main__":

    images_dir = Path("/home/tenzin3/ocr-ann-transfer/images_dir/W2PD17382-I1KG81275/")
    xml_dir = Path("/home/tenzin3/ocr-ann-transfer/images_dir/W2PD17382-I1KG81275/page")
    output_dir = Path("cropped_images")
    images_cropping_pipeline(images_dir, xml_dir, output_dir)
    
