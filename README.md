# HF Summarizer (Streamlit + CLI)

Summarize text, PDFs, DOCX, or URLs with Hugging Face transformers.

## 🚀 Local run (Windows/Mac/Linux)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧰 CLI examples
```bash
python summarize.py --file docs/my.pdf --out summary.txt
python summarize.py --url "https://example.com/article" --max-new 160
python summarize.py --text "Long paragraph here …"
```

## ☁️ Deploy on Streamlit Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io → New app.
3. Select this repo, branch `main`, and app file `app.py` → Deploy.

> Streamlit Cloud is CPU-only. For speed, select `sshleifer/distilbart-cnn-12-6` in the sidebar.

## 🔧 Notes
- GPU is auto-detected; if available, the pipeline uses half precision for speed.
- Very long docs are chunked and then compressed in a second pass.
- Optional: set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` on Windows to silence cache warnings.
