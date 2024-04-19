from pathlib import Path 

MISMATCH_LINE_TEXT_IMG_MAPPING = Path(".mismatch_line_map.txt")

def add_img_path_to_mismatch(image_path:str):
    if not MISMATCH_LINE_TEXT_IMG_MAPPING.exists():
        MISMATCH_LINE_TEXT_IMG_MAPPING.write_text(f"{image_path}\n")
        return
    with open(MISMATCH_LINE_TEXT_IMG_MAPPING,'a') as file:
        file.write(f"{image_path}\n")

CUR_DIR = Path(__file__).parent 
MAIN_DIR = CUR_DIR.parent.parent
MODEL_CONFIG_PATH = MAIN_DIR/ "model"/ "line_model_config.json"

print(MAIN_DIR)