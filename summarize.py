import argparse, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import trafilatura

MODEL_DEFAULT = "facebook/bart-large-cnn"

def device_info():
    use_cuda = torch.cuda.is_available()
    dev = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else None
    return dev, dtype, use_cuda

def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif suffix in {".docx"}:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def read_url(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError("Failed to fetch URL.")
    extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not extracted:
        raise ValueError("Failed to extract main text from URL.")
    return extracted

def chunk_by_tokens(text: str, tokenizer, max_tokens=900):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub = tokens[i:i+max_tokens]
        chunks.append(tokenizer.decode(sub))
    return chunks

def build_pipeline(model_id: str, device: int, dtype):
    kwargs = {}
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    tok = AutoTokenizer.from_pretrained(model_id)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
    return pipeline("summarization", model=mod, tokenizer=tok, device=device), tok

def summarize_long(text: str, model_id=MODEL_DEFAULT, max_new_tokens=120, second_pass=True):
    device, dtype, use_cuda = device_info()
    pipe, tok = build_pipeline(model_id, device, dtype)

    parts = chunk_by_tokens(text, tok, max_tokens=900)
    partial = pipe(parts, max_new_tokens=max_new_tokens//2, do_sample=False, truncation=True)
    stitched = " ".join(p["summary_text"] for p in partial)
    if second_pass:
        final = pipe(stitched, max_new_tokens=max_new_tokens, do_sample=False, truncation=True)
        return final[0]["summary_text"], use_cuda
    return stitched, use_cuda

def main():
    ap = argparse.ArgumentParser(description="Summarize text/file/URL with BART.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Raw text input")
    src.add_argument("--file", help="Path to .txt/.pdf/.docx")
    src.add_argument("--url", help="Web page URL")

    ap.add_argument("--model", default=MODEL_DEFAULT, help="HF model id")
    ap.add_argument("--max-new", type=int, default=140, help="Max NEW tokens generated")
    ap.add_argument("--no-second-pass", action="store_true", help="Disable compressing the stitched summary")
    ap.add_argument("--out", help="Write summary to this file")

    args = ap.parse_args()

    if args.text:
        text = args.text
    elif args.file:
        text = read_text(Path(args.file))
    else:
        text = read_url(args.url)

    t0 = time.time()
    summary, gpu = summarize_long(
        text,
        model_id=args.model,
        max_new_tokens=args.max_new,
        second_pass=not args.no_second_pass
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
