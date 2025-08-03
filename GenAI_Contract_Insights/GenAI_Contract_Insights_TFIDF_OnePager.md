# GenAI Contract Insights — TF‑IDF Approach (One Pager)

## Objective
Analyze contracts to extract key clauses, score risks, generate a concise summary, and answer questions — using a lightweight **TF‑IDF retriever** (no vector DB dependency).

## Architecture
- **UI:** Streamlit (file upload, tables, risk, summary, Q&A)
- **API:** FastAPI (`/extract`, `/ask`)
- **Parsing:** `pdfplumber`, `python-docx`; OCR stub for scanned PDFs
- **Extraction:** Regex-based clause detection (parties, effective date, termination, payment terms, liability)
- **Retrieval:** TF‑IDF + cosine similarity to select top‑K chunks as context
- **Generation:** Optional OpenAI LLM for summary/Q&A; fallback heuristic if no key
- **Risk Engine:** Rule-based (liability > $10M, payment > Net 45, auto-renewal)
- **Storage:** Local file system; no external services required

## Data Flow
1. Upload document → parsed to text and cleaned.
2. Text chunked by paragraphs (<= ~800–1200 chars).
3. Build TF‑IDF matrix over chunks; user query transformed and matched via cosine.
4. Top‑K contexts fed to LLM (if available) to reduce hallucination; otherwise heuristic response.
5. Clauses extracted via regex; risk scored; results rendered in UI and JSON.

## Strengths
- Zero external infra; quick to demo anywhere.
- Deterministic, fast retrieval; easy to explain.
- Works offline; consistent latency.

## Trade‑offs
- Lexical match only; misses semantic paraphrases.
- No persistence across sessions unless you re-index every time.
- Heuristic summary/Q&A quality is limited without LLM.

## Extensibility
- Add keyword expansion or BM25.
- Swap to vector DB later (e.g., Chroma/FAISS) with embeddings.
- Add OCR/Tesseract; persist outputs to SQLite/cloud.

## Security
- Stays on device by default; LLM calls only if OPENAI key set.
- Secrets loaded via `.env`; no keys in code.

## Demo Script (2–3 min)
1. Upload `msa_sample.txt` → show extracted table.
2. Point to risk flag (liability > $10M = High).
3. Show summary → mention LLM optional.
4. Ask: “What’s the termination notice period?” → reveal retrieved chunks.
