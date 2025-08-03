# GenAI Contract Insights & Risk Summarization (Chroma Edition)

An interview-ready, end-to-end demo that analyzes contracts:
- Ingestion (PDF/DOCX/TXT)
- Clause extraction (regex)
- **RAG with Chroma** (OpenAI or local Sentence-Transformers embeddings)
- Risk scoring (rule-based)
- Executive summary & Q&A
- Streamlit UI + FastAPI API

## Quick Start

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) (Optional) .env
Create a `.env` in project root:
```
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
# Or local embeddings (no API):
# ST_EMBED_MODEL=all-MiniLM-L6-v2
CHROMA_DIR=vectorstore/chromadb
```

### 3) Run Streamlit UI (recommended in interview)
```bash
streamlit run frontend/streamlit_app.py
```
Then upload a file from `data/sample_contracts/` and demo:
- Extracted clauses
- Risk summary
- Summary (LLM optional; uses retrieved context)
- Q&A with retrieved contexts

### 4) Or run the API
```bash
uvicorn app.main:app --reload --port 8000
```
Open Swagger at http://127.0.0.1:8000/docs

## Tech
- **UI:** Streamlit
- **API:** FastAPI
- **RAG:** ChromaDB (persistent)
- **Embeddings:** OpenAI (if key present) or Sentence-Transformers fallback
- **Parsing:** pdfplumber, python-docx
- **Risk:** rule-based thresholds

## Notes
- First run with local embeddings will download a small model.
- Data stays local unless you enable OpenAI.
- For scanned PDFs, integrate OCR later (stub provided).

## License
MIT
