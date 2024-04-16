import csv 
from pathlib import Path 

from ocr_ann_transfer.utility import sort_paths_and_get_strings

def create_page_texts(text:str, volume:str, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = text.split("$")
    for i, page in enumerate(pages, 1):
        file_name = f"{volume}_{i:05}.txt"
        Path(output_dir/file_name).write_text(page, encoding="utf-8")
    print("[SUCESS]: page texts successfully created.")



def map_page_texts_to_images(images_dir:Path, page_texts_dir:Path, output_file_path:Path=Path("page_texts_to_images.csv")):
    images = list(images_dir.rglob("*.jpg"))
    page_texts = list(page_texts_dir.rglob("*.txt"))

    if len(page_texts) != len(images):
        print("[ERROR]: Number of page texts and images are not equal!")
        return 
    
    images = sort_paths_and_get_strings(images)
    page_texts = sort_paths_and_get_strings(page_texts)

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
            page_text_path = Path(row['Line Text Paths'])
            image_folder, image_name = Path(output_dir/image_path.stem), image_path.stem
            image_folder.mkdir(parents=True, exist_ok=True)
            text = page_text_path.read_text(encoding="utf-8")
            line_texts = text.splitlines()
            for idx, line_text in enumerate(line_texts, start=1):
                Path(image_folder/f"{image_name}_{idx:04}.txt").write_text(line_text)

    print("[SUCESS]: line texts successfully created.")


def map_line_texts_to_images(cropped_images_dir:Path, line_texts_dir:Path, output_file_path:Path=Path("line_texts_to_images.csv")):
    images_subdir = list(cropped_images_dir.iterdir())
    line_texts_subdir = list(line_texts_dir.iterdir())

    missing_line_texts = 0
    mismatch_count = 0

    mapping_res = {"images":[], "texts":[]}
    for image_subdir in images_subdir:
        line_text_subdir = next((text for text in line_texts_subdir if text.name == image_subdir.name), None)
        if line_text_subdir:
            images = list(image_subdir.rglob("*.jpg"))
            line_texts = list(line_text_subdir.rglob("*.txt"))

            if len(images) != len(line_texts):
                mismatch_count += 1

            images = sort_paths_and_get_strings(images)
            line_texts = sort_paths_and_get_strings(line_texts)

            mapping_res["images"].extend(images)
            mapping_res["texts"].extend(line_texts)

        else:
            missing_line_texts += 1
    print(f"No of missing line texts subdir is {missing_line_texts}")
    print(f"No of mismatch is {mismatch_count}")

    
    """ write the correct results to csv"""
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Paths', 'Page Text Paths'])
        for image_path, line_texts_path in zip(mapping_res["images"], mapping_res["texts"]):
            writer.writerow([image_path, line_texts_path])





if __name__ == "__main__":
    cropped_images_dir = Path("cropped_images")
    line_texts_dir = Path("line_texts_dir")
    map_line_texts_to_images(cropped_images_dir, line_texts_dir)