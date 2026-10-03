import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

from core.llm import get_secret

# ── Settings ────────────────────────────────────────────────────────────────
VOXTRAL_URL = "https://api.mistral.ai/v1/audio/transcriptions"
VOXTRAL_MODEL = os.getenv("VOXTRAL_MODEL", "voxtral-mini-latest")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")   # small se tez

SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_PIECE_SECONDS = 25     # Sarvam sync API max 30s leta hai
SARVAM_WORKERS = 3            # itne pieces ek saath bheje jaayenge

_whisper_model = None


# ── English: Voxtral (Mistral API), fallback Whisper ────────────────────────
def transcribe_chunk_voxtral(chunk_path: str) -> str:
    api_key = get_secret("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY set nahi hai (.env ya Streamlit Secrets).")

    last_status = None
    for attempt in range(4):
        with open(chunk_path, "rb") as f:
            response = requests.post(
                VOXTRAL_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(chunk_path), f, "audio/wav")},
                data={"model": VOXTRAL_MODEL},
                timeout=180,
            )

        if response.status_code == 429:
            last_status = 429
            time.sleep(3 * (attempt + 1))
            continue

        if not response.ok:
            raise RuntimeError(f"Voxtral error {response.status_code}: {response.text}")

        return response.json().get("text", "").strip()

    raise RuntimeError(f"Voxtral rate limited ({last_status}) after retries.")


def _load_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper  # lazy import: torch tabhi load hoga jab zarurat ho
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
    return _whisper_model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    model = _load_whisper()
    result = model.transcribe(
        chunk_path,
        task="transcribe",
        fp16=False,                       # CPU par warning aur slowdown se bachao
        condition_on_previous_text=False,
    )
    return result["text"].strip()


# ── Hinglish: Sarvam (parallel pieces) ──────────────────────────────────────
def _send_to_sarvam(piece_path: str) -> str:
    api_key = get_secret("SARVAM_API_KEY")
    model = get_secret("SARVAM_STT_MODEL", "saaras:v2.5")
    if not api_key:
        raise RuntimeError("SARVAM_API_KEY set nahi hai (.env ya Streamlit Secrets).")

    headers = {"api-subscription-key": api_key}

    for attempt in range(3):
        with open(piece_path, "rb") as f:
            response = requests.post(
                SARVAM_STT_TRANSLATE_URL,
                headers=headers,
                files={"file": (os.path.basename(piece_path), f, "audio/wav")},
                data={"model": model, "with_diarization": "false"},
                timeout=120,
            )

        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))
            continue

        if not response.ok:
            raise RuntimeError(f"Sarvam error {response.status_code}: {response.text}")

        return response.json().get("transcript", "")

    raise RuntimeError("Sarvam API baar-baar fail ho raha hai. Thodi der baad try karo.")


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    piece_paths = []
    try:
        for i, start in enumerate(range(0, len(audio), piece_ms)):
            piece_path = f"{chunk_path}_sv_{i}.wav"
            audio[start:start + piece_ms].export(piece_path, format="wav")
            piece_paths.append(piece_path)

        print(f"  → Sarvam: {len(piece_paths)} piece(s) parallel bhej raha hoon ...")
        with ThreadPoolExecutor(max_workers=SARVAM_WORKERS) as pool:
            texts = list(pool.map(_send_to_sarvam, piece_paths))  # order same rehta hai

        return " ".join(t.strip() for t in texts if t).strip()
    finally:
        for p in piece_paths:
            if os.path.exists(p):
                os.remove(p)


# ── Router ──────────────────────────────────────────────────────────────────
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)

    try:
        return transcribe_chunk_voxtral(chunk_path)
    except Exception as voxtral_error:
        print(f"Voxtral fail hua ({voxtral_error}). Whisper par ja raha hoon ...")
        try:
            return transcribe_chunk_whisper(chunk_path)
        except Exception as whisper_error:
            raise RuntimeError(
                f"Voxtral failed: {voxtral_error} | Whisper fallback failed: {whisper_error}"
            )


def transcribe_all(chunks: list, language: str = "english") -> str:
    engine = "Sarvam AI" if language.lower() == "hinglish" else "Voxtral (Mistral)"
    print(f"Using {engine} for transcription.")

    parts = []
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        parts.append(transcribe_chunk(chunk, language=language))

    print("Transcription complete.")
    return " ".join(p for p in parts if p).strip()