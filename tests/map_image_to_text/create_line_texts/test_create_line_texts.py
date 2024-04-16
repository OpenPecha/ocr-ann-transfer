import csv 
from pathlib import Path 
import shutil 

from ocr_ann_transfer.map_image_to_text import create_line_texts


def adjust_file_paths_in_csv_file(original_csv_path:Path, base_dir):
    adjusted_csv_path = base_dir / "adjusted_page_texts_to_images.csv"

    with open(original_csv_path, 'r', newline='') as infile, \
         open(adjusted_csv_path, 'w', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        headers = next(reader)
        writer.writerow(headers)

        for row in reader:
            adjusted_row = [str(base_dir / Path(col)) for col in row]
            writer.writerow(adjusted_row)
    return adjusted_csv_path

def test_create_line_texts():
    DATA_DIR = Path(__file__).parent / "data"
    page_texts_to_image_mapping = DATA_DIR / "page_texts_to_images.csv"

    adjusted_csv_path = adjust_file_paths_in_csv_file(page_texts_to_image_mapping, DATA_DIR)
    output_dir = DATA_DIR / "line_texts"
    create_line_texts(adjusted_csv_path, output_dir)

    """ check if the line text for I1KG812750004 is created. """
    image_name = "I1KG812750004"
    line_text_dir = Path(output_dir/f"{image_name}")
    
    assert line_text_dir.exists()

    """ check if the file is properly divided into line texts files"""

    for i in range(1,5):
        assert Path(line_text_dir/f"{image_name}_{i:04}.txt").exists()
    
    """ deleting the output files and folder"""
    shutil.rmtree(output_dir)
    adjusted_csv_path.unlink()





test_create_line_texts()
