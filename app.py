# app.py
# Streamlit Hugging Face Summarizer (text / file / URL)
# Works on CPU (Streamlit Cloud safe). No CUDA required.

import io
import math
import time
from typing import List, Tuple

import streamlit as st

# Text + file handling
from pypdf import PdfReader
from docx import Document
import trafilatura

# HF inference
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name: str, device: int = -1):
    """
    Load tokenizer + model + pipeline once (cached).
    device=-1 -> CPU, 0 -> first GPU (we keep CPU for Streamlit Cloud).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,       # CPU on Streamlit Cloud
        framework="pt",
    )
    return tokenizer, summarizer


def read_uploaded_file(file) -> str:
    """Read text from uploaded .pdf/.docx/.txt/.md"""
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.read()))
        texts = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts).strip()
    elif name.endswith(".docx"):
        bio = io.BytesIO(file.read())
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs).strip()
    elif name.endswith(".txt") or name.endswith(".md"):
        return file.read().decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX, TXT, or MD.")


def fetch_url_text(url: str) -> str:
    """Fetch and extract readable text from a URL with trafilatura."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return ""
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False) or ""
    return extracted.strip()


def tokens_from_words_est(words: int) -> int:
    """
    Rough conversion for seq2seq lengths.
    Heuristic: ~1 word ≈ 1.3 tokens (English).
    """
    return max(1, int(words * 1.3))


def sliding_windows(tokens: List[int], window: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Produce (start, end) index pairs over token ids with overlap.
    """
    if window <= 0:
        return [(0, len(tokens))]
    if overlap >= window:
        overlap = max(0, window // 4)  # safety
    spans = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + window)
        spans.append((start, end))
        if end == n:
            break
        start = end - overlap
    return spans


def chunk_by_tokens(text: str, tokenizer, max_input_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Chunk a long text into token windows (with overlap) according to model limits.
    """
    enc = tokenizer(text, return_tensors=None, add_special_tokens=False)
    ids = enc["input_ids"]
    spans = sliding_windows(ids, max_input_tokens, overlap_tokens)
    chunks = []
    for s, e in spans:
        piece_ids = ids[s:e]
        piece = tokenizer.decode(piece_ids, skip_special_tokens=True)
        chunks.append(piece.strip())
    return chunks


def summarize_chunks(
    chunks: List[str],
    summarizer,
    target_words: int,
    min_words: int,
    gen_max_tokens_limit: int,
) -> List[str]:
    """
    Summarize each chunk with reasonable generation lengths.
    We try to target `target_words` per chunk (soft).
    """
    # Convert words to approx tokens for generation controls
    max_new_tokens = min(gen_max_tokens_limit, tokens_from_words_est(target_words))
    min_new_tokens = max(1, min(tokens_from_words_est(min_words), max_new_tokens // 2))

    outputs = []
    for i, ch in enumerate(chunks, 1):
        # HF summarization pipeline expects max_length/min_length in tokens of the OUTPUT text.
        # We use new length controls that map to generate() args when available.
        result = summarizer(
            ch,
            max_length=max_new_tokens,
            min_length=min_new_tokens,
            do_sample=False,
            truncation=True,
        )
        out = result[0]["summary_text"].strip()
        outputs.append(out)
    return outputs


def two_pass_summarize(
    text: str,
    tokenizer,
    summarizer,
    target_words: int,
    min_words: int,
    token_margin: int,
    overlap_tokens: int,
    second_pass: bool,
) -> str:
    """
    1) Chunk to model input size minus margin
    2) Summarize each chunk
    3) Optionally summarize the concatenation again to hit final target
    """
    # Model input context (e.g., 1024 for BART, 4096 for some longformer-like models)
    model_ctx = getattr(tokenizer, "model_max_length", 1024)
    # Keep a safety margin
    max_input_tokens = max(64, model_ctx - token_margin)

    # Break into input-sized pieces
    chunks = chunk_by_tokens(text, tokenizer, max_input_tokens, overlap_tokens)

    # First pass: per-chunk summaries
    # Generation max tokens cap – avoid runaway on small models
    gen_cap = 1024
    first_pass = summarize_chunks(
        chunks,
        summarizer,
        target_words=max(50, target_words),  # sensible floor
        min_words=min_words,
        gen_max_tokens_limit=gen_cap,
    )

    if not second_pass or len(first_pass) == 1:
        return "\n\n".join(first_pass).strip()

    # Second pass: compress the concatenated summaries down to about the requested target
    joined = "\n\n".join(first_pass)
    final = summarize_chunks(
        [joined],
        summarizer,
        target_words=max(100, target_words),  # slightly higher, model will compress anyway
        min_words=min_words,
        gen_max_tokens_limit=gen_cap,
    )
    return final[0].strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="HF Summarizer", page_icon="📝", layout="wide")
st.title("📝 Hugging Face Summarizer (Text • File • URL)")

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.text_input(
        "Model (Hugging Face)",
        value="facebook/bart-large-cnn",
        help="Any seq2seq summarization model from Hugging Face Hub (e.g., 'facebook/bart-large-cnn', 'google/pegasus-xsum').",
    )
    target_words = st.number_input("Target words per summary chunk", min_value=30, max_value=2000, value=180, step=10)
    min_words = st.number_input("Minimum words per chunk summary", min_value=10, max_value=1000, value=60, step=10)
    token_margin = st.number_input(
        "Token margin (input safety buffer)",
        min_value=16, max_value=2048, value=128, step=16,
        help="Subtract this from the model's max input length before chunking, to avoid truncation."
    )
    overlap_tokens = st.number_input(
        "Overlap tokens between chunks",
        min_value=0, max_value=2048, value=64, step=16,
        help="Adds context continuity between adjacent chunks."
    )
    second_pass = st.checkbox(
        "Enable second pass (summarize the summaries)",
        value=True,
        help="After summarizing each chunk, run a final pass to produce a tighter overall summary.",
    )
    show_debug = st.checkbox("Show debug info", value=False)

# Input selection tabs
t1, t2, t3 = st.tabs(["✍️ Text", "📄 File", "🔗 URL"])

source_text = ""
source_kind = None

with t1:
    txt = st.text_area("Enter or paste text", height=220, placeholder="Paste the text you want to summarize…")
    if st.button("Summarize Text", key="summ_text"):
        if not txt.strip():
            st.warning("Please enter some text.")
        else:
            source_text = txt.strip()
            source_kind = "text"

with t2:
    up = st.file_uploader("Upload a PDF / DOCX / TXT / MD", type=["pdf", "docx", "txt", "md"])
    if st.button("Summarize File", key="summ_file"):
        if not up:
            st.warning("Please upload a file.")
        else:
            try:
                source_text = read_uploaded_file(up)
                source_kind = f"file: {up.name}"
            except Exception as e:
                st.error(f"Failed to read file: {e}")

with t3:
    url = st.text_input("Enter a URL")
    if st.button("Summarize URL", key="summ_url"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and extracting page content…"):
                text = fetch_url_text(url.strip())
            if not text:
                st.error("Could not extract readable text from this URL.")
            else:
                source_text = text
                source_kind = f"url: {url.strip()}"

# Main action
if source_text:
    colA, colB = st.columns([1, 2])
    with colA:
        st.subheader("Source")
        st.caption(f"Source: {source_kind}")
        st.write(source_text[:2000] + ("…" if len(source_text) > 2000 else ""))

    with colB:
        st.subheader("Summary")
        try:
            with st.spinner(f"Loading model: {model}"):
                tokenizer, summarizer = load_pipeline(model)

            start = time.time()
            summary = two_pass_summarize(
                text=source_text,
                tokenizer=tokenizer,
                summarizer=summarizer,
                target_words=int(target_words),
                min_words=int(min_words),
                token_margin=int(token_margin),
                overlap_tokens=int(overlap_tokens),
                second_pass=bool(second_pass),
            )
            elapsed = time.time() - start
            st.success(f"Done in {elapsed:.1f}s")
            st.write(summary)

            st.download_button(
                "⬇️ Download summary (.txt)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain",
            )

        except Exception as e:
            st.error("App error while summarizing. See details below.")
            st.exception(e)

    if show_debug:
        st.divider()
        st.subheader("🔍 Debug")
        try:
            ctx = getattr(tokenizer, "model_max_length", None)
        except Exception:
            ctx = None
        st.write(
            {
                "model": model,
                "tokenizer.model_max_length": ctx,
                "settings": {
                    "target_words": target_words,
                    "min_words": min_words,
                    "token_margin": token_margin,
                    "overlap_tokens": overlap_tokens,
                    "second_pass": second_pass,
                },
                "source_kind": source_kind,
                "source_length_chars": len(source_text),
            }
        )

# Footer hint
st.caption(
    "Tip: For very long documents, increase overlap a bit (e.g., 64–128) and keep a healthy token margin (≥128) "
    "so inputs don’t get truncated by the model’s max sequence length."
)
