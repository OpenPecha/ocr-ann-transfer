import csv 
from pathlib import Path 
from typing import List, Optional

from ocr_ann_transfer.utility import sort_paths_and_get_strings, get_wronged_cropped_images, get_largest_image_paths
from ocr_ann_transfer.config import add_img_path_to_mismatch

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
            page_text_path = Path(row['Page Text Paths'])
            image_folder, image_name = Path(output_dir/image_path.stem), image_path.stem
            image_folder.mkdir(parents=True, exist_ok=True)
            text = page_text_path.read_text(encoding="utf-8")
            line_texts = text.splitlines()
            for idx, line_text in enumerate(line_texts, start=1):
                Path(image_folder/f"{image_name}_{idx:04}.txt").write_text(line_text)

    print("[SUCESS]: line texts successfully created.")


def map_line_texts_to_images(cropped_images_dir:Path, line_texts_dir:Path, wronged_cropped_images:Optional[List[Path]]=[]):
    """ mapping would be saved here"""
    output_file_path:Path=Path("line_texts_to_images.csv")

    images_subdir = list(cropped_images_dir.iterdir())
    line_texts_subdir = list(line_texts_dir.iterdir())

    missing_line_texts = 0
    more_texts = 0
    empty_images= 0

    images_subdir.sort(key=lambda x: x.name)
    mapping_res = {"images":[], "texts":[]}
    for image_subdir in images_subdir:
        line_text_subdir = next((text for text in line_texts_subdir if text.name == image_subdir.name), None)
        if line_text_subdir:
            images = list(image_subdir.rglob("*.jpg"))
            line_texts = list(line_text_subdir.rglob("*.txt"))
            if len(line_texts)==0 or len(images)==0:
                empty_images += 1
                continue
            """ if number of cropped images is more than line texts, try filtering out the images"""
            filtered_images = []
            if len(images) != len(line_texts):
                filtered_images = [image for image in images if image not in wronged_cropped_images]
                if len(filtered_images) == 0:
                    empty_images += 1
                """ if there are more line texts, nothing to do."""
                if len(images) < len(line_texts):
                    msg = f"{str(image_subdir)}, images: {len(images)}, texts: {len(line_texts)}"
                    add_img_path_to_mismatch(msg)
                    more_texts += 1
                    continue
                """ if there filtering was success, carry on with it."""
                if len(filtered_images) == len(line_texts):
                    images = filtered_images
                else:   
                    """ if there more line images than line texts, and filtering was unsuccessful, """
                    """ then match with number of line texts, by sorting with size"""
                    images = get_largest_image_paths(filtered_images, len(line_texts))

            images = sort_paths_and_get_strings(images)
            line_texts = sort_paths_and_get_strings(line_texts)
            if len(images) == len(line_texts):
                mapping_res["images"].extend(images)
                mapping_res["texts"].extend(line_texts)

        else:
            missing_line_texts += 1
    print(f"No of missing line texts subdir is {missing_line_texts}")
    print(f"No of mismatches (more line texts than line images) is {more_texts}")
    print(f"No of empty images is {empty_images}")
    """ write the correct results to csv"""
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Paths', 'Line Text Paths'])
        for image_path, line_texts_path in zip(mapping_res["images"], mapping_res["texts"]):
            writer.writerow([image_path, line_texts_path])





if __name__ == "__main__":
    # text = Path("new_v003.txt").read_text(encoding="utf-8").replace("\n\n","$")
    # create_page_texts(text, "3",Path("page_texts_dir"))
    
    # images_dir = Path("images/W2PD17382-I1KG81275")
    # page_texts_dir = Path("page_texts_dir")
    # map_page_texts_to_images(images_dir, page_texts_dir)

    # create_line_texts(Path("page_texts_to_images.csv"), Path("line_texts_dir"))
    cropped_images_dir = Path("ocr_output")
    line_texts_dir = Path("line_texts_dir")
    wronged_cropped_images = get_wronged_cropped_images("1")
    map_line_texts_to_images(cropped_images_dir, line_texts_dir, wronged_cropped_images)