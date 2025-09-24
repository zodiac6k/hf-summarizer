import io
import time
import requests
import torch
import streamlit as st

from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pypdf import PdfReader
from docx import Document
from readability import Document as ReadabilityDoc
from lxml import html as lxml_html
import trafilatura


# =========================
# Page & constants
# =========================
st.set_page_config(page_title="HF Summarizer", page_icon="📝", layout="wide")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
MAX_WORDS = 12000  # soft cap for input length to avoid OOM on cloud CPU
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"  # fast default (CPU-friendly)
HEAVY_MODEL = "facebook/bart-large-cnn"          # higher quality, slower (best w/ GPU)


# =========================
# Utilities
# =========================
def device_info():
    on_gpu = torch.cuda.is_available()
    device = 0 if on_gpu else -1
    dtype = torch.float16 if on_gpu else None
    return device, dtype, on_gpu


def read_pdf(file) -> str:
    reader = PdfReader(file)
    return "\n".join([p.extract_text() or "" for p in reader.pages])


def read_docx(file) -> str:
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_url_text(url: str) -> str:
    """
    Robust URL text extraction:
    1) Try trafilatura (fast, high quality)
    2) Fallback to requests + readability-lxml with a desktop User-Agent
    """
    # 1) Try trafilatura
    try:
        downloaded = trafilatura.fetch_url(url, with_metadata=False, no_ssl=True, timeout=15)
        txt = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
            deduplicate=True,
        )
        if txt and len(txt.strip()) > 200:
            return txt
    except Exception:
        pass

    # 2) Fallback: requests + readability
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20, allow_redirects=True)
    r.raise_for_status()
    readable = ReadabilityDoc(r.text)
    article_html = readable.summary(html_partial=False)
    body = lxml_html.fromstring(article_html).text_content()
    body = " ".join(body.split())

    if not body or len(body) < 200:
        raise ValueError("Couldn't extract sufficient article text (site may block bots or content is very short).")
    return body


@st.cache_resource(show_spinner=False)
def load_pipeline(model_id: str, device: int, dtype):
    """
    Cache HF pipeline + tokenizer. If GPU, load model in half precision to save VRAM.
    """
    kw = {}
    if dtype is not None:
        kw["torch_dtype"] = dtype
    tok = AutoModelForSeq2SeqLM.from_pretrained  # just to keep IDE hints happy
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)
    pipe = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)
    return pipe, tokenizer


def chunk_by_tokens(text: str, tokenizer, max_tokens=900):
    """
    Token-aware chunking so we never hit model max length (BART limit ~1024 tokens).
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(ids[i:i + max_tokens]) for i in range(0, len(ids), max_tokens)]


def summarize_chunks(pipe, chunks, max_new_tokens=96, beams=1):
    """
    Batch summarization for speed (especially on CPU). Greedy/low-beam improves latency.
    """
    batch_size = 4
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=beams,             # 1 = greedy (fast), 2 = small quality bump
        no_repeat_ngram_size=3,
        early_stopping=True,
        truncation=True,
    )
    outputs = []
    with torch.inference_mode():
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            res = pipe(batch, **gen_kwargs)
            outputs.extend([r["summary_text"] for r in res])
    return outputs


def summarize_text(text: str, model_id: str, max_new: int, allow_second_pass: bool):
    """
    Main entry: token-aware chunking, batched generation, optional second pass on GPU only.
    """
    device, dtype, on_gpu = device_info()
    pipe, tok = load_pipeline(model_id, device, dtype)

    parts = chunk_by_tokens(text, tok, max_tokens=900)

    # Progress UI
    status = st.empty()
    prog = st.progress(0.0)

    partial = []
    beams = 2 if on_gpu else 1  # use a tiny bit more search only when GPU is available
    for i in range(len(parts)):
        status.info(f"Summarizing chunk {i + 1}/{len(parts)} …")
        # do half budget per chunk to keep each piece concise
        out = summarize_chunks(pipe, [parts[i]], max_new_tokens=max_new // 2, beams=beams)[0]
        partial.append(out)
        prog.progress((i + 1) / len(parts))

    status.empty()
    prog.empty()

    stitched = " ".join(partial)

    # Second pass only when GPU is present (faster & better), skip on CPU for speed
    if allow_second_pass and on_gpu and stitched.strip():
        stitched = summarize_chunks(pipe, [stitched], max_new_tokens=max_new, beams=2)[0]

    return stitched, on_gpu


# =========================
# UI
# =========================
st.title("📝 Hugging Face Summarizer")
st.caption("Fast, robust summarization for Text • PDF/DOCX • URLs — GPU auto-detect, batching, and token-aware chunking.")

with st.sidebar:
    st.header("Settings")
    model_id = st.selectbox(
        "Model",
        options=[DEFAULT_MODEL, HEAVY_MODEL, "facebook/bart-base"],
        index=0,  # default to DistilBART for speed on CPU/cloud
    )
    max_new = st.slider("Max new tokens (per pass)", 32, 256, 128, 8)
    second_pass = st.checkbox("Second pass compression (GPU-only recommended)", value=True)

    device, _, on_gpu = device_info()
    st.write(f"**Device:** {'GPU ✅' if on_gpu else 'CPU'}")

tab_text, tab_file, tab_url = st.tabs(["✍️ Text", "📄 File (PDF/TXT/DOCX)", "🔗 URL"])

input_text = ""

with tab_text:
    input_text = st.text_area(
        "Paste text to summarize",
        height=220,
        placeholder="Paste your long text here…",
    )

with tab_file:
    up = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
    if up is not None:
        try:
            if up.type == "application/pdf":
                input_text = read_pdf(up)
            elif up.type in ("text/plain",):
                input_text = up.read().decode("utf-8", errors="ignore")
            else:
                input_text = read_docx(up)
            st.success(f"Loaded {len(input_text.split())} words from file.")
        except Exception as e:
            st.error(f"File read failed: {e}")

with tab_url:
    url = st.text_input("Enter webpage URL")
    if url and st.button("Fetch URL"):
        with st.spinner("Fetching & extracting main content…"):
            try:
                input_text = fetch_url_text(url)
                st.success(f"Fetched {len(input_text.split())} words.")
            except Exception as e:
                st.error(f"URL extraction failed: {e}")

# Soft cap very large inputs to keep latency reasonable on CPU/cloud
if input_text and len(input_text.split()) > MAX_WORDS:
    st.warning(f"Input is large ({len(input_text.split())} words). "
               f"I’ll process the first {MAX_WORDS} words.")
    input_text = " ".join(input_text.split()[:MAX_WORDS])

col1, col2 = st.columns([1, 1])
with col1:
    if input_text:
        st.caption(f"Input words: {len(input_text.split())}")

if st.button("Summarize", type="primary"):
    if not input_text or len(input_text.strip()) < 10:
        st.warning("Please provide some content first.")
    else:
        t0 = time.time()
        with st.spinner("Generating summary…"):
            try:
                summary, used_gpu = summarize_text(input_text, model_id, max_new, second_pass)
            except Exception as e:
                st.error(f"Error during summarization: {e}")
                summary = ""
        dt = time.time() - t0

        if summary:
            st.success(f"Done in {dt:.2f}s on {'GPU' if used_gpu else 'CPU'}.")
            st.subheader("Summary")
            st.write(summary)

            # word counts
            st.caption(f"Summary words: {len(summary.split())}")

            # Downloads: TXT / MD / DOCX
            buf_txt = io.BytesIO(summary.encode("utf-8"))
            st.download_button("Download .txt", buf_txt, file_name="summary.txt")

            md = f"# Summary\n\n_Model: {model_id} • Device: {'GPU' if used_gpu else 'CPU'} • Time: {dt:.2f}s_\n\n{summary}\n"
            buf_md = io.BytesIO(md.encode("utf-8"))
            st.download_button("Download .md", buf_md, file_name="summary.md")

            doc = Document()
            doc.add_heading('Summary', 0)
            meta = f"Model: {model_id} | Device: {'GPU' if used_gpu else 'CPU'} | Time: {dt:.2f}s"
            doc.add_paragraph(meta)
            doc.add_paragraph(summary)
            buf_docx = io.BytesIO()
            doc.save(buf_docx)
            buf_docx.seek(0)
            st.download_button("Download .docx", buf_docx, file_name="summary.docx")
