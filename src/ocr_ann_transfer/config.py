from pathlib import Path 
MISMATCH_LINE_TEXT_IMG_MAPPING = Path(".mismatch_line_map.txt")

def add_img_path_to_mismatch(image_path:str):
    if not MISMATCH_LINE_TEXT_IMG_MAPPING.exists():
        MISMATCH_LINE_TEXT_IMG_MAPPING.write_text(f"{image_path}\n")
        return
    with open(MISMATCH_LINE_TEXT_IMG_MAPPING,'a') as file:
        file.write(f"{image_path}\n")