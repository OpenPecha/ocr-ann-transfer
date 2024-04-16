from pathlib import Path

from ocr_ann_transfer.ann_transfer import transfer_page_annotations 


def test_transfer_page_annotations():
    DATA_DIR = Path(__file__).parent / "data"
    source_text = Path(DATA_DIR / "ocr_text.txt").read_text(encoding="utf-8").replace("\n\n","$")
    target_text = Path(DATA_DIR/ "clean_text.txt").read_text(encoding="utf-8").replace("\n\n","")
    output_text = transfer_page_annotations(source_text, target_text).replace("$","\n\n")

    expected_text = Path(DATA_DIR/ "expected.txt").read_text(encoding="utf-8")
    assert output_text == expected_text


