import os
import time

import httpx
from langchain_mistralai import ChatMistralAI

# Mistral free tier ~1 request/second. Is gap se 429 bahut kam aata hai.
# Chaho to .env me MISTRAL_PAUSE_SECONDS=2 likh kar badal sakte ho.
PAUSE_SECONDS = float(os.getenv("MISTRAL_PAUSE_SECONDS", "1.5"))


def pause(seconds: float | None = None) -> None:
    """Do LLM calls ke beech chhota gap, rate limit se bachne ke liye."""
    time.sleep(PAUSE_SECONDS if seconds is None else seconds)


def get_llm(temperature: float = 0.3):
    """
    ChatMistralAI jo 429 (rate limit) aane par khud ruk kar dobara try karta hai.
    Wait time: ~1s, 2s, 4s, 8s... (jitter ke saath), max 6 attempts.
    """
    llm = ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
    )

    return llm.with_retry(
        retry_if_exception_type=(httpx.HTTPStatusError, httpx.TransportError),
        wait_exponential_jitter=True,
        stop_after_attempt=6,
    )