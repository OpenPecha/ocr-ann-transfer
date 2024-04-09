from pathlib import Path 


def create_page_texts(text:str, volume:str, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pages = text.split("$")
    for i, page in enumerate(pages, 1):
        Path(output_dir/f"{volume}_{i}.txt").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    file_path = Path("new_v003.txt")
    text = file_path.read_text(encoding="utf-8").replace("\n\n", "$")
    output_dir = Path("page_texts")

    create_page_texts(text, "3", output_dir)
