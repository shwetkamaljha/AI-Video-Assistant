import os
import time
import threading

from langchain_mistralai import ChatMistralAI

PRIMARY_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
FALLBACK_MODEL = os.getenv("MISTRAL_FALLBACK_MODEL", "ministral-8b-2512")

MIN_GAP_SECONDS = 1.2        # do calls ke beech minimum gap
MAX_TRIES_PER_MODEL = 3      # ek model par kitni baar retry

_lock = threading.Lock()
_last_call = 0.0


def get_secret(name: str, default=None):
    """Pehle environment/.env se, phir Streamlit secrets se key padho."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def _throttle():
    global _last_call
    with _lock:
        wait = MIN_GAP_SECONDS - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limited" in text
        or '"code":"1300"' in text
    )


def _retry_after(error: Exception, default: float) -> float:
    try:
        value = error.response.headers.get("retry-after")
        return min(float(value), 20) if value else default
    except Exception:
        return default


def get_llm(temperature: float = 0.3, model: str = None):
    return ChatMistralAI(
        model=model or PRIMARY_MODEL,
        mistral_api_key=get_secret("MISTRAL_API_KEY"),
        temperature=temperature,
        max_retries=1,
    )


def call_llm(messages, temperature: float = 0.3) -> str:
    """
    messages: [("system", "..."), ("human", "...")]
    Throttle + 429 par retry + fallback model. Text (str) return karta hai.
    """
    last_error = None

    for model in (PRIMARY_MODEL, FALLBACK_MODEL):
        llm = get_llm(temperature, model)

        for attempt in range(MAX_TRIES_PER_MODEL):
            _throttle()
            try:
                content = llm.invoke(messages).content
                if isinstance(content, list):
                    content = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                return content.strip()
            except Exception as e:
                last_error = e
                if _is_rate_limit(e):
                    time.sleep(_retry_after(e, 2 * (attempt + 1)))
                    continue
                raise  # 401, 400 jaise errors retry se theek nahi hote

    raise RuntimeError(
        f"Mistral rate limit baar-baar aa raha hai. Thodi der baad try karo. ({last_error})"
    ) from last_error