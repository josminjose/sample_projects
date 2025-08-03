from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os, shutil, tempfile

from .ocr_module import extract_text
from .extract_clauses import extract
from .risk_engine import score
from .rag_pipeline import summarize, answer_question

app = FastAPI(title="GenAI Contract Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        clauses = extract(text)
        risks = score(clauses, text)
        summary = summarize(text)
        return {"clauses": clauses, "risk": risks, "summary": summary, "text_len": len(text)}
    finally:
        os.unlink(tmp_path)

@app.post("/ask")
async def ask_endpoint(question: str = Form(...), file: UploadFile = File(...)) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = extract_text(tmp_path)
        resp = answer_question(question, text)
        return resp
    finally:
        os.unlink(tmp_path)
