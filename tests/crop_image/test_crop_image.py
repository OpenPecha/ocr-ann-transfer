from pathlib import Path 
import shutil 

from ocr_ann_transfer.crop_image import images_cropping_pipeline


def test_crop_image():
    DATA_DIR = Path(__file__).parent / "data"
    images_dir = DATA_DIR / "image"
    xml_dir = DATA_DIR / "xml"
    output_dir = DATA_DIR / "cropped_images"
    images_cropping_pipeline(images_dir, xml_dir, output_dir)

    """ I1KG812750004.jpg is in test case """
    image_name = "I1KG812750004"

    """ check if a folder for this image is created """
    cropped_image_dir = Path(output_dir / f"{image_name}")
    assert cropped_image_dir.exists()
    
    """ check if image is properly cropped to 4 line images"""
    for i in range(1,5):
        assert Path(cropped_image_dir/f"{image_name}_cropped_{i}.jpg").exists()

    shutil.rmtree(output_dir)

