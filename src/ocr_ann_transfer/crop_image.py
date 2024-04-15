from pathlib import Path 
from typing import List, Tuple
import xml.etree.ElementTree as ET
from PIL import Image

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
    output_dir.mkdir(parents=True, exist_ok=True)
    images_folder = output_dir/ image_path.stem
    images_folder.mkdir(parents=True, exist_ok=True)

    for i, coords in enumerate(coords_list, start=1):
        cropped_img = crop_image(image_path, coords)
        if cropped_img:
            cropped_img_path = images_folder / f"{str(image_path).rsplit('.', 1)[0]}_cropped_{i}.jpg"
            cropped_img.save(cropped_img_path)
            print(f"Cropped image saved to: {cropped_img_path}")
        else:
            print("Skipping saving due to cropping error.")

if __name__ == "__main__":
    xml_path = Path('I1KG812750003.xml')  
    image_path = Path("I1KG812750003.jpg") 
    coords_list = parse_page_xml(xml_path)

    output_dir = Path("cropped_images")
    crop_multiple_images(image_path, coords_list, output_dir)
