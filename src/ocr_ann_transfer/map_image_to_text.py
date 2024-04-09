import csv 
from pathlib import Path 
from typing import List 

def create_page_texts(text:str, volume:str, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = text.split("$")
    for i, page in enumerate(pages, 1):
        file_name = f"{volume}_{i:05}.txt"
        Path(output_dir/file_name).write_text(page, encoding="utf-8")


def sort_paths_by_string(paths:List[Path]) -> List[str]:
    """ sort paths by string and return as string"""
    return [str(path) for path in sorted(paths, key=lambda path: str(path))]

def map_page_texts_to_images(images_dir:Path, page_texts_dir:Path, output_file_path:Path=Path("page_texts_to_images.csv")):
    images = list(images_dir.rglob("*.jpg"))
    page_texts = list(page_texts_dir.rglob("*.txt"))

    if len(page_texts) != len(images):
        print("[ERROR]: Number of page texts and images are not equal!")
        return 
    
    images = sort_paths_by_string(images)
    page_texts = sort_paths_by_string(page_texts)

    """ write to csv """
    with open(output_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['Image Paths', 'Page Text Paths'])
        for img_path, text_path in zip(images, page_texts):
            csvwriter.writerow([img_path, text_path])


if __name__ == "__main__":
    images_dir = Path("images/W2PD17382-I1KG81275")
    page_texts_dir = Path("page_texts")

    map_page_texts_to_images(images_dir, page_texts_dir)
