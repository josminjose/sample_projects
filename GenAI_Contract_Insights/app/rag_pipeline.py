from __future__ import annotations
from typing import List, Dict, Tuple
import re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .prompts import SUMMARY_PROMPT, QA_PROMPT
import os

from dotenv import load_dotenv
# Load variables from .env into the environment
load_dotenv()

def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur: chunks.append(cur)
            cur = p
    if cur: chunks.append(cur)
    return chunks

def build_index(chunks: List[str]):
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(chunks)
    return vec, X

def retrieve(query: str, vec, X, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X)[0]
    idxs = sims.argsort()[::-1][:k]
    return [(chunks[i], float(sims[i])) for i in idxs]

def _call_openai(prompt: str, context: str, instruction: str) -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI()
        sys = instruction
        user = f"""Context:\n{context}\n\nTask:\n{prompt}"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def summarize(text: str) -> str:
    chunks = chunk_text(text, max_chars=1200)
    ctx = "\n---\n".join(chunks[:3])
    llm = _call_openai("Summarize the contract as requested.", ctx, SUMMARY_PROMPT)
    if llm:
        return llm
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = "• " + "\n• ".join(lines[:6])[:800]
    return "Executive Summary (heuristic):\n" + head

def answer_question(question: str, text: str) -> Dict:
    chunks = chunk_text(text)
    vec, X = build_index(chunks)
    hits = retrieve(question, vec, X, chunks, k=3)
    ctx = "\n---\n".join([h[0] for h in hits])
    llm = _call_openai(f"Answer the question: {question}", ctx, QA_PROMPT)
    ans = llm if llm else "Based on the retrieved context, I don't have enough information to answer with certainty."
    return {"answer": ans, "contexts": hits}
