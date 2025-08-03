from __future__ import annotations
from .utils import read_text_from_file, clean_text

def extract_text(file_path: str) -> str:
    raw = read_text_from_file(file_path)
    return clean_text(raw)
