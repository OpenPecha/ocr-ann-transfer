from pathlib import Path 
from ocr_ann_transfer.map_image_to_text import create_page_texts 

def test_create_page_texts():
    DATA_DIR = Path(__file__).parent / "data"
    text = Path(DATA_DIR/ "test_file.txt").read_text(encoding="utf-8").replace("\n\n", "$")
    
    volume = "3"
    output_dir = DATA_DIR
    create_page_texts(text, volume, output_dir)

    page_freq = text.count("$")
    for i in range(page_freq+1):
        output_file = Path(DATA_DIR/ f"{volume}_{i+1}.txt")
        assert output_file.exists()
        output_file.unlink()
        

