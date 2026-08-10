# 🎬 AI Video Assistant

Turn any meeting recording, YouTube video, or local audio/video file into a **transcript, summary, action items, key decisions, and open questions** — then **chat with it** using a RAG-powered assistant.

---

## 🧩 The Problem It Solves

People sit through long meetings and videos, but:
- Nobody has time to rewatch a 45-minute call to find what was decided
- Action items and owners get lost in a wall of speech
- Follow-up questions never get tracked
- Searching *"what did we say about the launch date?"* means scrubbing through a video timeline manually

**AI Video Assistant** automates this end-to-end: give it a YouTube link, a local file, or an uploaded recording (English or Hinglish), and it turns raw speech into structured, searchable meeting intelligence — including a chat interface to ask follow-up questions directly against the transcript.

---

## ✨ Features

- 🔗 **Two input modes** — paste a YouTube URL, or upload a local audio/video file
- 🌐 **Bilingual transcription** — English (via local Whisper) and Hinglish (via Sarvam AI, which transcribes *and* translates to English in one step)
- 🏷️ **Auto-generated title** for every session
- 📋 **Structured meeting summary** (map-reduce summarization for long transcripts)
- ✅ **Action items** — task, owner, and deadline extracted automatically
- 🔑 **Key decisions** made during the meeting
- ❓ **Open questions** / unresolved follow-ups
- 💬 **RAG-based chat** — ask natural-language questions about the meeting; answers are grounded strictly in the transcript (no hallucinated context)
- 🖥️ **Interactive Streamlit UI** with live pipeline status

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Audio/video ingestion** | `yt-dlp` (YouTube download), `pydub` + `ffmpeg` (format conversion, chunking) |
| **Speech-to-Text (English)** | OpenAI Whisper (runs locally via PyTorch) |
| **Speech-to-Text (Hinglish)** | Sarvam AI Speech-to-Text-Translate API |
| **LLM orchestration** | LangChain (LCEL — chained runnables) |
| **LLM** | Mistral AI (`mistral-small-latest`) via `langchain-mistralai` |
| **Summarization strategy** | Map-reduce (chunk → summarize each → combine into final summary) |
| **Vector store (RAG)** | ChromaDB, persisted locally |
| **Embeddings** | HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`, CPU) |
| **Text splitting** | LangChain `RecursiveCharacterTextSplitter` |
| **Env/config** | `python-dotenv` |

---

## 🏗️ How It Works (Pipeline)

```
Input (YouTube URL / uploaded file)
        │
        ▼
1. Audio Processing        → download/convert to WAV (16kHz, mono), split into chunks
        │
        ▼
2. Transcription            → Whisper (English) or Sarvam AI (Hinglish → English)
        │
        ▼
3. Title Generation         → Mistral LLM generates a short session title
        │
        ▼
4. Summarization             → map-reduce: per-chunk summaries → combined final summary
        │
        ▼
5. Extraction                → action items, key decisions, open questions (parallel LLM chains)
        │
        ▼
6. RAG Engine Build          → transcript is chunked, embedded, and stored in ChromaDB
        │
        ▼
7. Chat                      → user question → similarity search (top-k) → context-grounded LLM answer
```

---

## 📁 Project Structure

```
.
├── app.py                     # Streamlit UI (entry point for the web app)
├── main.py                    # CLI entry point (run the pipeline from the terminal)
├── requirements.txt           # Python dependencies
├── packages.txt                # System-level dependency (ffmpeg) — for cloud deployment
├── .gitignore
├── utils/
│   └── audio_processor.py     # YouTube download, format conversion, chunking
└── core/
    ├── transcriber.py         # Whisper + Sarvam transcription logic
    ├── summarizer.py          # Title + summary generation (Mistral, map-reduce)
    ├── extractor.py           # Action items / decisions / questions extraction
    ├── rag_engine.py          # RAG chain construction + question answering
    └── vector_store.py        # ChromaDB vector store setup
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
> Also make sure **ffmpeg** is installed on your system (`sudo apt install ffmpeg` on Linux, or via Homebrew/Chocolatey on Mac/Windows).

### 3. Set up environment variables
Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
WHISPER_MODEL=small
SARVAM_STT_MODEL=saaras:v2.5
```

### 4. Run the app

**Web UI:**
```bash
streamlit run app.py
```

**CLI:**
```bash
python main.py
```

---

## ☁️ Deployment

This app is deployable on **Hugging Face Spaces** or **Streamlit Community Cloud**.
- `packages.txt` installs `ffmpeg` automatically on supported platforms
- API keys must be set as platform **secrets** (never commit `.env`)
- Whisper + PyTorch make this a memory-heavy app — Hugging Face Spaces is recommended over the Streamlit Cloud free tier

---

## ⚠️ Notes

- Local files are processed via a temporary path — nothing is permanently stored unless you add persistence yourself
- The RAG chat only answers from the meeting transcript — it will say so explicitly if the answer isn't present
- Sarvam's sync API caps audio at 30s per request; the transcriber automatically slices Hinglish audio into 25s pieces before sending

---

## 📄 License

Add your preferred license here (MIT, Apache 2.0, etc.)
