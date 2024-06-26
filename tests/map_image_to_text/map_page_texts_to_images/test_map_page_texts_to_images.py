from pathlib import Path 

from ocr_ann_transfer.map_image_to_text import map_page_texts_to_images


def test_map_page_texts_to_images():
    DATA_DIR = Path(__file__).parent / "data"
    images_dir = DATA_DIR / "images"
    page_texts_dir = DATA_DIR / "page_texts"

    output_file_path = DATA_DIR / "mapping.csv"
    map_page_texts_to_images(images_dir, page_texts_dir, output_file_path)
    assert output_file_path.exists()
    output_file_path.unlink()

