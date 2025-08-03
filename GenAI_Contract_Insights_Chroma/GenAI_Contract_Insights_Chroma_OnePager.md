# GenAI Contract Insights — Chroma RAG Approach (One Pager)

## Objective
Deliver robust, semantic retrieval for contract insights using **ChromaDB** with embeddings (OpenAI or local Sentence‑Transformers), enabling better recall and persistence.

## Architecture
- **UI:** Streamlit (file upload, tables, risk, summary, Q&A)
- **API:** FastAPI (`/extract`, `/ask`)
- **Parsing:** `pdfplumber`, `python-docx`; OCR stub
- **Extraction:** Regex-based clauses
- **RAG:** Chroma PersistentClient with collection per demo/app; kNN query over embeddings
- **Embeddings:** OpenAI (`text-embedding-3-small`) when `OPENAI_API_KEY` is set; otherwise local `all-MiniLM-L6-v2`
- **Generation:** Optional OpenAI chat (`gpt-4o-mini`); fallback heuristic
- **Risk Engine:** Rule-based thresholds
- **Storage:** Chroma persist dir (`vectorstore/chromadb`) + samples/outputs

## Data Flow
1. Upload → extract & clean text.
2. Chunk (~1200 chars, 120 overlap) → upsert into Chroma with metadata.
3. For summary: query a broad prompt to fetch representative chunks; feed to LLM or heuristic.
4. For Q&A: query collection with user question → top‑K chunks → LLM answer or heuristic fallback.
5. Combine with regex extraction and risk scoring → return to UI/API.

## Strengths
- Semantic recall across paraphrases and varied wording.
- Persistent index across requests/sessions.
- Swappable embedding backends (cloud or local).

## Trade‑offs
- First‑run model download (local) or API dependency (OpenAI).
- Slightly more setup than TF‑IDF.
- Vector hygiene needed if mixing multiple documents/users (collections, ids).

## Extensibility
- Add re‑ranking (e.g., Cross‑Encoder) for improved precision.
- Add doc‑level dedup and chunk‑linking.
- Multi‑tenant collections; audit logging.

## Security
- Local by default; embeddings may call remote if OpenAI enabled.
- Secrets via `.env`; configurable `CHROMA_DIR`.

## Demo Script (2–3 min)
1. Upload `msa_sample.txt` → auto‑index into Chroma.
2. Show summary and emphasize semantic retrieval.
3. Ask: “What is the termination notice period?” → show retrieved contexts from Chroma.
4. Mention persistence between calls and optional OpenAI usage.
