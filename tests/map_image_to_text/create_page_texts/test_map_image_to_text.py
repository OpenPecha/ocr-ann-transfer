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
        file_name = f"{volume}_{i+1:05}.txt"
        output_file = Path(DATA_DIR/ file_name)
        assert output_file.exists()
        output_file.unlink()
        

