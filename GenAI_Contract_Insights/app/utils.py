from __future__ import annotations
import re, os, io, pdfplumber
from docx import Document

def read_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".md"]:
        return open(path, "r", encoding="utf-8", errors="ignore").read()
    if ext in [".docx"]:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    if ext in [".pdf"]:
        out = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        return "\n".join(out)
    # default fallback
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def clean_text(t: str) -> str:
    t = t.replace("\x0c", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n\n", t)
    return t.strip()
