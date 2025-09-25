#!/usr/bin/env python3
import argparse, sys, time
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pypdf import PdfReader
from docx import Document
import trafilatura


MODEL_DEFAULT = "facebook/bart-large-cnn"
SENTINEL_TOO_LARGE = 32000  # if tokenizer.model_max_length is absurdly large


# ---------------- Device helpers ----------------
def device_info() -> Tuple[int, object, bool]:
    """Return (device_index, torch_dtype_or_None, using_cuda_bool)."""
    use_cuda = torch.cuda.is_available()
    dev = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else None
    return dev, dtype, use_cuda


# ---------------- I/O helpers ----------------
def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def read_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Failed to fetch URL.")
    extracted = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if not extracted:
        raise ValueError("Failed to extract main text from URL.")
    return extracted


# ---------------- Token helpers ----------------
def approx_tokens_from_words(words: int) -> int:
    # ~1.3 tokens/word for English (rough heuristic)
    return max(8, int(words * 1.3))


def get_model_ctx(tokenizer: AutoTokenizer) -> int:
    ctx = getattr(tokenizer, "model_max_length", 1024)
    # Some tokenizers set an absurd sentinel for "infinite"
    if ctx is None or ctx > SENTINEL_TOO_LARGE:
        ctx = 1024
    return int(ctx)


def chunk_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    max_chunk_tokens: int,
    overlap_tokens: int = 48,
) -> List[str]:
    """Chunk text by token count with optional overlap. Decodes with skip_special_tokens."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids)
    if n == 0:
        return []

    step = max(1, max_chunk_tokens - overlap_tokens)
    chunks = []
    i = 0
    while i < n:
        j = min(i + max_chunk_tokens, n)
        chunk_ids = ids[i:j]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text.strip())
        if j == n:
            break
        i += step
    return chunks


# ---------------- Pipeline ----------------
def build_pipeline(model_id: str, device: int, dtype):
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
    pipe = pipeline("summarization", model=mod, tokenizer=tok, device=device)
    return pipe, tok


# ---------------- Summarization core ----------------
def compute_lengths_for_text(
    text: str,
    tokenizer: AutoTokenizer,
    target_new_tokens: int,
    min_new_tokens: int,
) -> Tuple[int, int]:
    """Make per-input safe generation lengths; avoid max>input warnings."""
    in_len = len(tokenizer.encode(text, add_special_tokens=False))
    # Bound new tokens proportionally to input length
    # Aim ~60% of input tokens but not above target
    max_new = min(target_new_tokens, max(16, int(0.6 * in_len)))
    min_new = min(min_new_tokens, max(8, int(0.2 * in_len)))
    if min_new >= max_new:
        min_new = max(8, max_new // 2)
    return max_new, min_new


def summarize_chunks_batched(
    pipe,
    chunks: List[str],
    tokenizer: AutoTokenizer,
    target_new_tokens: int,
    min_new_tokens: int,
    batch_size: int = 4,
) -> List[str]:
    """Batch summarize chunks while adapting gen lengths per chunk."""
    if not chunks:
        return []

    # Compute per-chunk lengths
    per_item = []
    for ch in chunks:
        mx, mn = compute_lengths_for_text(ch, tokenizer, target_new_tokens, min_new_tokens)
        per_item.append(dict(text=ch, max_new_tokens=mx, min_length=mn))

    # The pipeline API doesn't accept per-item params easily,
    # so we process in mini-batches with uniform lengths within the batch
    # by computing safe batch-level params (min over max_new_tokens; min over min_length).
    outs = []
    for i in range(0, len(per_item), batch_size):
        batch = per_item[i:i+batch_size]
        # Use conservative lengths across the batch
        batch_max_new = min(x["max_new_tokens"] for x in batch)
        batch_min_len = min(x["min_length"] for x in batch)
        texts = [x["text"] for x in batch]
        res = pipe(
            texts,
            max_new_tokens=batch_max_new,
            min_length=batch_min_len,
            do_sample=False,
            truncation=True,  # safety belt; we already chunk to avoid encoder overflow
        )
        for r in res:
            outs.append(r["summary_text"].strip())
    return outs


def summarize_long(
    text: str,
    model_id: str = MODEL_DEFAULT,
    token_margin: int = 64,
    overlap_tokens: int = 48,
    target_words: int = 120,
    min_words: int = 30,
    second_pass: bool = True,
) -> Tuple[str, bool]:
    device, dtype, use_cuda = device_info()
    pipe, tok = build_pipeline(model_id, device, dtype)

    model_ctx = get_model_ctx(tok)
    max_chunk_tokens = max(128, model_ctx - int(token_margin))

    # Convert word-intentions to token counts for new tokens
    target_new_tokens = approx_tokens_from_words(target_words)
    min_new_tokens = approx_tokens_from_words(min_words)

    # Fast-path for short inputs
    in_tok_len = len(tok.encode(text, add_special_tokens=False))
    if in_tok_len <= max_chunk_tokens:
        mx, mn = compute_lengths_for_text(text, tok, target_new_tokens, min_new_tokens)
        res = pipe(text, max_new_tokens=mx, min_length=mn, do_sample=False, truncation=True)
        return res[0]["summary_text"].strip(), use_cuda

    # Long input → chunk
    chunks = chunk_by_tokens(text, tok, max_chunk_tokens=max_chunk_tokens, overlap_tokens=overlap_tokens)
    partial = summarize_chunks_batched(
        pipe,
        chunks,
        tok,
        target_new_tokens=target_new_tokens // 2,  # first pass shorter
        min_new_tokens=min_new_tokens // 2,
        batch_size=4,
    )
    stitched = " ".join(partial).strip()

    if second_pass and stitched:
        mx2, mn2 = compute_lengths_for_text(
            stitched,
            tok,
            target_new_tokens=max(64, int(target_new_tokens * 0.8)),
            min_new_tokens=max(24, int(min_new_tokens * 0.8)),
        )
        final = pipe(stitched, max_new_tokens=mx2, min_length=mn2, do_sample=False, truncation=True)
        return final[0]["summary_text"].strip(), use_cuda

    return stitched, use_cuda


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Summarize text/file/URL (robust, chunked, 2-pass).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Raw text input")
    src.add_argument("--file", help="Path to .txt/.pdf/.docx")
    src.add_argument("--url", help="Web page URL")

    ap.add_argument("--model", default=MODEL_DEFAULT, help="HF model id (Seq2Seq summarizer)")
    ap.add_argument("--target-words", type=int, default=120, help="Target words per summary output")
    ap.add_argument("--min-words", type=int, default=30, help="Minimum words per summary output")
    ap.add_argument("--token-margin", type=int, default=64, help="Safety margin below model context")
    ap.add_argument("--overlap-tokens", type=int, default=48, help="Overlap tokens between chunks")
    ap.add_argument("--no-second-pass", action="store_true", help="Disable second-pass compression")
    ap.add_argument("--out", help="Write summary to this file")

    args = ap.parse_args()

    # Load source text
    if args.text:
        text = args.text
    elif args.file:
        text = read_text(Path(args.file))
    else:
        text = read_url(args.url)

    if not text or not text.strip():
        print("No text extracted to summarize.", file=sys.stderr)
        sys.exit(2)

    t0 = time.time()
    summary, gpu = summarize_long(
        text=text,
        model_id=args.model,
        token_margin=args.token_margin,
        overlap_tokens=args.overlap_tokens,
        target_words=args.target_words,
        min_words=args.min_words,
        second_pass=not args.no_second_pass,
    )
    dt = time.time() - t0

    header = f"## Device: {'GPU' if gpu else 'CPU'} | Model: {args.model} | Time: {dt:.2f}s"
    out = f"{header}\n\n{summary}"
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"Saved to {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
