import time, io
import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pypdf import PdfReader
from docx import Document
import trafilatura

st.set_page_config(page_title="Summarizer", page_icon="📝", layout="wide")

# ---------- Helpers ----------
def device_info():
    use_cuda = torch.cuda.is_available()
    return (0 if use_cuda else -1), (torch.float16 if use_cuda else None), use_cuda

def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Failed to fetch URL.")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted:
        raise ValueError("Failed to extract main text.")
    return extracted

@st.cache_resource(show_spinner=False)
def load_pipe(model_id: str, device: int, dtype):
    kw = {}
    if dtype is not None:
        kw["torch_dtype"] = dtype
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)
    return pipeline("summarization", model=mod, tokenizer=tok, device=device), tok

def chunk_by_tokens(text: str, tokenizer, max_tokens=900):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(sub))
    return chunks

def summarize_text(text: str, model_id: str, max_new: int, do_second_pass: bool):
    device, dtype, on_gpu = device_info()
    pipe, tok = load_pipe(model_id, device, dtype)

    parts = chunk_by_tokens(text, tok, max_tokens=900)
    partial = []
    status = st.empty()
    pb = st.progress(0.0)
    for i, chunk in enumerate(parts, 1):
        status.info(f"Summarizing chunk {i}/{len(parts)} …")
        s = pipe(chunk, max_new_tokens=max_new//2, do_sample=False, truncation=True)[0]["summary_text"]
        partial.append(s)
        pb.progress(i/len(parts))
    pb.empty(); status.empty()

    stitched = " ".join(partial)
    if do_second_pass and stitched.strip():
        stitched = pipe(stitched, max_new_tokens=max_new, do_sample=False, truncation=True)[0]["summary_text"]

    return stitched, on_gpu

# ---------- UI ----------
st.title("📝 Hugging Face Summarizer")
st.caption("BART-based summarization with smart chunking, GPU auto-detect, and exports.")

with st.sidebar:
    st.header("Settings")
    model_id = st.selectbox(
        "Model",
        options=[
            "facebook/bart-large-cnn",
            "sshleifer/distilbart-cnn-12-6",  # faster on CPU
            "facebook/bart-base"
        ],
        index=0
    )
    max_new = st.slider("Max new tokens", 32, 256, 128, 8)
    do_second_pass = st.checkbox("Second pass compression", value=True)

    device, _, on_gpu = device_info()
    st.write(f"**Device:** {'GPU ✅' if on_gpu else 'CPU'}")

tab_text, tab_file, tab_url = st.tabs(["✍️ Text", "📄 File (PDF/TXT/DOCX)", "🔗 URL"])

input_text = ""
with tab_text:
    input_text = st.text_area("Paste text to summarize", height=220, placeholder="Paste your long text here…")

with tab_file:
    up = st.file_uploader("Upload PDF/TXT/DOCX", type=["pdf", "txt", "docx"])
    if up is not None:
        if up.type == "application/pdf":
            input_text = read_pdf(up)
        elif up.type in ("text/plain",):
            input_text = up.read().decode("utf-8", errors="ignore")
        else:
            input_text = read_docx(up)

with tab_url:
    url = st.text_input("Enter webpage URL")
    if url and st.button("Fetch URL"):
        with st.spinner("Fetching and extracting main content…"):
            try:
                input_text = read_url(url)
                st.success("Content extracted.")
            except Exception as e:
                st.error(str(e))

if st.button("Summarize", type="primary"):
    if not input_text or len(input_text.strip()) < 10:
        st.warning("Please provide some content first.")
    else:
        t0 = time.time()
        with st.spinner("Generating summary…"):
            try:
                summary, on_gpu = summarize_text(input_text, model_id, max_new, do_second_pass)
            except Exception as e:
                st.error(f"Error: {e}")
                summary = ""
        dt = time.time() - t0

        if summary:
            st.success(f"Done in {dt:.2f}s on {'GPU' if on_gpu else 'CPU'}.")
            st.subheader("Summary")
            st.write(summary)

            # Downloads
            buf_txt = io.BytesIO(summary.encode("utf-8"))
            st.download_button("Download .txt", buf_txt, file_name="summary.txt")

            md = f"# Summary\n\n_Model: {model_id} • Device: {'GPU' if on_gpu else 'CPU'} • Time: {dt:.2f}s_\n\n{summary}\n"
            buf_md = io.BytesIO(md.encode("utf-8"))
            st.download_button("Download .md", buf_md, file_name="summary.md")
