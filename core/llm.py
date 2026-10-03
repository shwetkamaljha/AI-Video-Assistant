import os
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter

# Saare modules ek hi limiter share karenge: max ~1 request / 2 seconds
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)


def get_llm(temperature: float = 0.3):
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        rate_limiter=_rate_limiter,
        max_retries=6,
    )