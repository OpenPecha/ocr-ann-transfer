from pathlib import Path

from OCR_non_bdrc.create_text_from_OCR import (
    get_text_from_OCR as extract_texts_from_OCR,
)


def get_text_from_ocr(ocr_path: Path):
    page_texts = extract_texts_from_OCR(ocr_path)
    volume_text = ""

    page_break = "\n\n"
    page_texts_items = list(page_texts.items())
    for page_no, page_text_dict in page_texts_items:
        page_text = page_text_dict["text"]
        """ normalizing page break """
        if page_text is None:  # image with no text
            page_text = page_break
        if page_no == len(page_texts_items):
            if page_text.endswith(page_break):
                page_text = page_text.replace(page_break, "")
        else:
            page_text = page_text.rstrip()
            page_text += page_break
        volume_text += page_text
    return volume_text


if __name__ == "__main__":
    ocr_path = Path("OCR/W2PD17382-I1KG81275/")
    texts = get_text_from_ocr(ocr_path)
    Path("ocr_v003.txt").write_text(texts, encoding="utf-8")
