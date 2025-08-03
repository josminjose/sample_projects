from __future__ import annotations
from typing import List, Dict
import os, uuid
import chromadb
from chromadb.utils import embedding_functions
from .prompts import SUMMARY_PROMPT, QA_PROMPT

from dotenv import load_dotenv
load_dotenv()


def get_embedding_fn():
    if os.getenv("OPENAI_API_KEY"):
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        )
    else:
        model = os.getenv("ST_EMBED_MODEL", "all-MiniLM-L6-v2")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)

PERSIST_DIR = os.getenv("CHROMA_DIR", "vectorstore/chromadb")

def get_collection(name: str = "contracts"):
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    emb_fn = get_embedding_fn()
    return client.get_or_create_collection(name=name, embedding_function=emb_fn)

def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
                tail = cur[-overlap:]
                cur = (tail + "\n\n" + p) if overlap and tail else p
            else:
                chunks.append(p[:max_chars])
                cur = p[max_chars-overlap:]
    if cur:
        chunks.append(cur)
    return chunks

def index_document(text: str, doc_id: str | None = None, collection_name: str = "contracts") -> Dict:
    doc_id = doc_id or str(uuid.uuid4())
    chunks = chunk_text(text)
    col = get_collection(collection_name)
    ids = [f"{doc_id}::chunk::{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
    col.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return {"doc_id": doc_id, "num_chunks": len(chunks)}

def retrieve(query: str, top_k: int = 4, where: Dict | None = None, collection_name: str = "contracts"):
    col = get_collection(collection_name)
    res = col.query(query_texts=[query], n_results=top_k, where=where)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = []
    for d, m, dist in zip(docs, metas, dists):
        score = None if dist is None else max(0.0, 1.0/(1.0+float(dist)))
        hits.append({"text": d, "meta": m, "score": score})
    return hits

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
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=350,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""

def build_index_and_summarize(text: str, collection_name: str = "contracts") -> str:
    index_document(text, collection_name=collection_name)
    hits = retrieve("overall summary of key terms", top_k=3, collection_name=collection_name)
    ctx = "\n---\n".join([h["text"] for h in hits]) if hits else text[:2000]
    llm = _call_openai("Summarize the contract as requested.", ctx, SUMMARY_PROMPT)
    if llm:
        return llm
    lines = [l.strip() for l in (ctx or text).splitlines() if l.strip()]
    head = "• " + "\n• ".join(lines[:6])[:800]
    return "Executive Summary (heuristic):\n" + head

def answer_question(question: str, collection_name: str = "contracts") -> Dict:
    hits = retrieve(question, top_k=4, collection_name=collection_name)
    ctx = "\n---\n".join([h["text"] for h in hits]) if hits else ""
    llm = _call_openai(f"Answer the question: {question}", ctx, QA_PROMPT)
    ans = llm if llm else "Based on the retrieved context, I don't have enough information to answer with certainty."
    return {"answer": ans, "contexts": [(h["text"], h["score"] if h["score"] is not None else 1.0) for h in hits]}
