from pathlib import Path

from ocr_ann_transfer.text_from_ocr import get_text_from_ocr


def test_get_text_from_ocr():
    DATA_DIR = Path(__file__).parent / "data"
    ocr_path = DATA_DIR / "W2PD17382-I1KG81275"
    output_text = get_text_from_ocr(ocr_path)
    expected_text = Path(DATA_DIR / "expected_text.txt").read_text(encoding="utf-8")
    assert expected_text == output_text

