from pathlib import Path 

from antx.core import transfer, get_diffs


def transfer_page_annotations(source_text:str, target_text:str):
    
    page_break_ann = "$"
    """ returns a diff with 3 types of strings: 0 overlaps, 1 target and -1 source. """
    diffs = get_diffs(source_text, target_text)

    new_text = ""
    for diff in diffs:
        diff_res, diff_text = diff 
        if diff_res in [0,1]:
            new_text += diff_text
            continue
        if diff_res == -1 and page_break_ann in diff_text:
            page_break_freq = diff_text.count(page_break_ann)
            new_text += page_break_ann*page_break_freq
    return new_text


if __name__ == "__main__":
    source_text = Path(f"ocr_v003.txt").read_text(encoding="utf-8")
    source_text = source_text.replace("\n\n", "$")
    target_text = Path(f"clean_v003.txt").read_text(encoding="utf-8")
    target_text = target_text.replace("\n\n", "")
    new_text = transfer_page_annotations(source_text, target_text)
    Path("new_v003.txt").write_text(new_text.replace("$", "\n\n"), encoding="utf-8")
    