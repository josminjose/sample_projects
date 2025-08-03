import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile, os

from app.ocr_module import extract_text
from app.extract_clauses import extract
from app.risk_engine import score
from app.rag_pipeline import summarize, answer_question

st.set_page_config(page_title="GenAI Contract Insights", layout="wide")
st.title("🔎 GenAI Contract Insights & Risk Summarization")

st.sidebar.header("Demo Steps")
st.sidebar.markdown("""
1. Upload a sample contract
2. View extracted clauses
3. Check risk summary
4. Generate executive summary
5. Ask contract-specific questions
""")

uploaded = st.file_uploader("Upload a contract (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

sample_dir = Path(__file__).resolve().parents[1] / "data" / "sample_contracts"
st.sidebar.subheader("Sample Contracts")
for p in sample_dir.glob("*.*"):
    st.sidebar.write(f"• {p.name}")

if uploaded:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    text = extract_text(tmp_path)
    clauses = extract(text)
    risks = score(clauses, text)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Extracted Clauses")
        if clauses:
            st.table(pd.DataFrame([clauses]).T.rename(columns={0: "Value"}))
        else:
            st.info("No clauses detected by the simple patterns. Try a different file.")

    with col2:
        st.subheader("Risk Summary")
        st.metric("Overall Risk", risks.get("level", "Unknown"))
        findings = risks.get("findings", [])
        if findings:
            st.write(pd.DataFrame(findings))
        else:
            st.write("No specific risks flagged.")

    st.subheader("Executive Summary")
    with st.spinner("Generating summary..."):
        summary = summarize(text)
    st.write(summary)

    st.subheader("Ask a Question")
    q = st.text_input("Example: What is the termination notice period?")
    if q:
        with st.spinner("Retrieving context and answering..."):
            ans = answer_question(q, text)
        st.write("**Answer**:", ans["answer"])
        with st.expander("Show retrieved contexts"):
            for i, (ctx, score) in enumerate(ans["contexts"], 1):
                st.markdown(f"**Chunk {i} (score {score:.3f})**\n\n{ctx}")
else:
    st.info("Upload a contract to start. Or use the samples listed in the sidebar.")
