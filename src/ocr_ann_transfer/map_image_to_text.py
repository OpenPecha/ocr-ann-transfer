import csv 
from pathlib import Path 
from typing import List 

def create_page_texts(text:str, volume:str, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = text.split("$")
    for i, page in enumerate(pages, 1):
        file_name = f"{volume}_{i:05}.txt"
        Path(output_dir/file_name).write_text(page, encoding="utf-8")
    print("[SUCESS]: page texts successfully created.")


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

    print(f"[SUCESS]: page texts to images mapped at {str(output_file_path)}")
    return output_file_path


def create_line_texts(page_texts_to_image_mapping:Path, output_dir:Path):

    output_dir.mkdir(parents=True, exist_ok=True)
    with page_texts_to_image_mapping.open(mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_path = Path(row['Image Paths'])
            page_text_path = Path(row['Page Text Paths'])
            image_folder, image_name = Path(output_dir/image_path.stem), image_path.stem
            image_folder.mkdir(parents=True, exist_ok=True)
            text = page_text_path.read_text(encoding="utf-8")
            line_texts = text.splitlines()
            for idx, line_text in enumerate(line_texts):
                Path(image_folder/f"{image_name}_{idx:04}.txt").write_text(line_text)

    print("[SUCESS]: line texts successfully created.")
if __name__ == "__main__":
    create_line_texts(Path("page_texts_to_images.csv"), Path("line_texts_dir"))