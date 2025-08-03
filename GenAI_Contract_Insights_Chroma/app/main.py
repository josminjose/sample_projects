from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os, tempfile

from dotenv import load_dotenv
load_dotenv()

from .ocr_module import extract_text
from .extract_clauses import extract
from .risk_engine import score
from .rag_chroma import build_index_and_summarize, answer_question as answer_with_chroma

app = FastAPI(title="GenAI Contract Insights API (Chroma)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        clauses = extract(text)
        risks = score(clauses, text)
        summary = build_index_and_summarize(text, collection_name="contracts_api")
        return {"clauses": clauses, "risk": risks, "summary": summary, "text_len": len(text)}
    finally:
        os.unlink(tmp_path)

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        # ensure indexing for this request
        _ = build_index_and_summarize(text, collection_name="contracts_api")
        resp = answer_with_chroma(question, collection_name="contracts_api")
        return resp
    finally:
        os.unlink(tmp_path)
