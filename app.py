# app.py
import math
from typing import List, Tuple

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# ------------ UI CONFIG ------------
st.set_page_config(page_title="HF URL/Text Summarizer", layout="wide")
st.title("📝 HF Summarizer (Robust, Chunked, 2-Pass)")

with st.sidebar:
    st.header("⚙️ Settings")

    # Solid default; feel free to add more models here
    model_name = st.selectbox(
        "Model",
        [
            "facebook/bart-large-cnn",   # 1024 tok ctx
            "google/pegasus-xsum",       # ~1k tok ctx (varies by tokenizer)
            "t5-large",                  # 512 tok ctx by default
        ],
        index=0,
        help="Choose a Seq2Seq summarization model.",
    )

    # Target words for final summary; we'll map to tokens dynamically per chunk
    target_words = st.slider(
        "Target words per chunk",
        min_value=40, max_value=220, value=120, step=10,
        help="Used to set max_length per chunk (approx)."
    )
    min_words = st.slider(
        "Minimum words per chunk",
        min_value=10, max_value=120, value=30, step=5,
        help="Lower bound for chunk summaries."
    )

    # Chunk parameters
    token_margin = st.number_input(
        "Token safety margin (per chunk)",
        min_value=16, max_value=200, value=64, step=8,
        help="We keep chunks under (model_max_length - margin) to avoid overflow."
    )
    overlap_tokens = st.number_input(
        "Overlap tokens between chunks",
        min_value=0, max_value=256, value=48, step=8,
        help="Small overlap improves coherence across chunk boundaries."
    )

    do_second_pass = st.checkbox(
        "Second-pass summarize (recommended for long texts)",
        value=True,
        help="Summarize chunk summaries again for a tighter final result."
    )

    do_sample = st.checkbox(
        "Enable sampling (more creative, less deterministic)",
        value=False
    )
    temperature = st.slider(
        "Temperature (if sampling)",
        min_value=0.0, max_value=2.0, value=1.0, step=0.1
    )
    top_p = st.slider(
        "Top-p (if sampling)",
        min_value=0.0, max_value=1.0, value=0.9, step=0.05
    )

    st.caption("No deprecated flags like `early_stopping` or pipeline-level `length_penalty` are used.")


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer, model


def get_device() -> int:
    # Streamlit Cloud usually runs CPU, but this handles local GPU too
    if torch.cuda.is_available():
        return 0
    return -1


def approx_tokens_from_words(words: int) -> int:
    # Rough heuristic: ~1.3 tokens/word (English). Adjust if you want.
    return max(8, int(words * 1.3))


def chunk_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_chunk_tokens: int,
    overlap_tokens: int
) -> List[str]:
    # Tokenize once
    ids = tokenizer.encode(text, truncation=False)
    chunks = []
    n = len(ids)
    i = 0
    step = max(1, max_chunk_tokens - overlap_tokens)

    while i < n:
        j = min(i + max_chunk_tokens, n)
        chunk_ids = ids[i:j]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if j == n:
            break
        i += step

    return chunks


def summarize_chunk(
    pipe,
    text: str,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    min_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    # Keep max_length < input length for short inputs (addresses your log warnings)
    # Compute input token length:
    in_tok_len = len(tokenizer.encode(text, truncation=False))

    # Ensure sensible bounds
    computed_max = min(max_tokens, max(16, int(0.6 * in_tok_len)))
    computed_min = min(min_tokens, max(8, int(0.2 * in_tok_len)))

    # Avoid min >= max which can crash some models
    if computed_min >= computed_max:
        computed_min = max(8, computed_max // 2)

    gen_kwargs = {
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update({"temperature": float(temperature), "top_p": float(top_p)})

    out = pipe(
        text,
        max_length=computed_max,
        min_length=computed_min,
        **gen_kwargs,
    )
    # pipeline sometimes returns list of dicts with key 'summary_text'
    if isinstance(out, list) and len(out) and "summary_text" in out[0]:
        return out[0]["summary_text"].strip()
    # Fallback
    return str(out)


def robust_summarize(
    text: str,
    model_name: str,
    target_words: int,
    min_words: int,
    token_margin: int,
    overlap_tokens: int,
    do_second_pass: bool,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[str, List[str]]:
    tokenizer, model = load_model_and_tokenizer(model_name)
    device = get_device()

    # Build pipeline (no deprecated flags like `early_stopping` / `length_penalty`)
    pipe = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,  # -1 = CPU, 0 = first GPU
    )

    model_ctx = getattr(tokenizer, "model_max_length", 1024)
    # Some tokenizers return very large sentinel values for "no limit" (e.g. 1000000000000000019884624838656)
    if model_ctx > 32000:
        model_ctx = 1024  # safe default if tokenizer has no meaningful max

    # Chunk size target (avoid overflow vs model limit)
    max_chunk_tokens = max(128, int(model_ctx - token_margin))

    # Convert user word targets to token targets (approx)
    max_tokens_per_chunk_summary = approx_tokens_from_words(target_words)
    min_tokens_per_chunk_summary = approx_tokens_from_words(min_words)

    # Short text fast-path
    in_tok_len = len(tokenizer.encode(text, truncation=False))
    if in_tok_len <= max_chunk_tokens:
        single = summarize_chunk(
            pipe,
            text,
            tokenizer,
            max_tokens=max_tokens_per_chunk_summary,
            min_tokens=min_tokens_per_chunk_summary,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return single, [single]

    # Long text → chunk then summarize each
    chunks = chunk_by_tokens(text, tokenizer, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
    chunk_summaries = []
    for i, ch in enumerate(chunks, 1):
        with st.status(f"Summarizing chunk {i}/{len(chunks)}...", expanded=False):
            s = summarize_chunk(
                pipe,
                ch,
                tokenizer,
                max_tokens=max_tokens_per_chunk_summary,
                min_tokens=min_tokens_per_chunk_summary,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            chunk_summaries.append(s)

    # Optionally, second pass on concatenated chunk summaries
    if do_second_pass:
        joined = "\n".join(chunk_summaries)
        # Aim shorter on the second pass
        second_pass_target = max(approx_tokens_from_words(int(target_words * 0.8)), 64)
        second_pass_min = max(approx_tokens_from_words(int(min_words * 0.8)), 24)
        final = summarize_chunk(
            pipe,
            joined,
            tokenizer,
            max_tokens=second_pass_target,
            min_tokens=second_pass_min,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return final, chunk_summaries

    # Otherwise return concatenated chunk summaries as final
    final = "\n".join(chunk_summaries)
    return final, chunk_summaries


# ------------ APP BODY ------------
tab1, tab2 = st.tabs(["Summarize Text", "Summarize URL (paste text after fetching)"])

with tab1:
    text = st.text_area(
        "Paste text to summarize",
        height=260,
        placeholder="Paste a long article, PDF text, or any content here...",
    )
    if st.button("Summarize", type="primary", disabled=not text.strip()):
        with st.spinner("Running summarization..."):
            try:
                final, parts = robust_summarize(
                    text=text,
                    model_name=model_name,
                    target_words=target_words,
                    min_words=min_words,
                    token_margin=token_margin,
                    overlap_tokens=overlap_tokens,
                    do_second_pass=do_second_pass,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                )
                st.subheader("✅ Final Summary")
                st.write(final)

                if len(parts) > 1:
                    with st.expander("Chunk summaries (first pass)"):
                        for i, p in enumerate(parts, 1):
                            st.markdown(f"**Chunk {i}**")
                            st.write(p)
                            st.markdown("---")
            except Exception as e:
                st.error(f"Summarization failed: {e}")


with tab2:
    st.write(
        "If you fetched a web page's content elsewhere, paste the raw text in the first tab. "
        "This app focuses on robust summarization with chunking."
    )
    st.info(
        "Tip: For PDFs, extract text (e.g., with `pypdf`) and paste into the **Summarize Text** tab."
    )
