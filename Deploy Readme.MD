# AI Video Assistant — Deploy Guide

## Local run
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Hugging Face Spaces (recommended — handles Whisper/Torch/ChromaDB well, free)
1. Go to https://huggingface.co/new-space
2. Space SDK: **Streamlit**
3. Push these files to the Space repo (via git or the web upload):
   - app.py
   - main.py (if used)
   - requirements.txt
   - packages.txt   → installs ffmpeg automatically
   - utils/ and core/ folders
   - .gitignore (do NOT push .env)
4. In Space Settings → **Repository secrets**, add:
   - MISTRAL_API_KEY
   - SARVAM_API_KEY
   - (any other keys your .env has)
5. Space builds automatically. First build takes 5–10 min (downloads Whisper model).

## Deploy on Streamlit Community Cloud (simpler, but free tier RAM may struggle)
1. Push code to a GitHub repo (without .env)
2. Go to https://share.streamlit.io → New app → pick repo, branch, and app.py
3. Advanced settings → Secrets → paste your keys in TOML format:
   ```
   MISTRAL_API_KEY = "..."
   SARVAM_API_KEY = "..."
   ```
4. Deploy. If it crashes on memory, switch to Hugging Face Spaces instead.