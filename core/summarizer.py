import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.llm import call_llm

MAX_DIRECT_CHARS = 15000


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    chunks = split_transcript(transcript)

    chunk_summaries = [
        call_llm(
            [
                ("system", "Summarize this portion of a meeting transcript concisely."),
                ("human", chunk),
            ],
            temperature=0.3,
        )
        for chunk in chunks
    ]

    combined = "\n\n".join(chunk_summaries)

    return call_llm(
        [
            (
                "system",
                "You are an expert meeting summarizer. Combine these partial summaries "
                "into one final professional meeting summary in bullet points.",
            ),
            ("human", combined),
        ],
        temperature=0.3,
    )


def generate_title(transcipt: str) -> str:
    return call_llm(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", transcipt[:2000]),
        ],
        temperature=0.3,
    )


# ── Sab kuch ek hi call mein ────────────────────────────────────────────────

def _condense(transcript: str) -> str:
    """Bahut lamba transcript ho to pehle chhota karo (tasks, owners, decisions bachaate hue)."""
    parts = []
    for chunk in split_transcript(transcript):
        parts.append(
            call_llm(
                [
                    (
                        "system",
                        "Summarize this portion of a meeting transcript. Keep names, tasks, "
                        "owners, deadlines, decisions and open questions.",
                    ),
                    ("human", chunk),
                ],
                temperature=0.2,
            )
        )
    return "\n\n".join(parts)


def _parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1], strict=False)
        except Exception:
            return None
    return None


def analyze_all(transcript: str) -> dict:
    """Title, summary, action items, decisions, questions: ek hi LLM call mein."""
    source = transcript
    if len(source) > MAX_DIRECT_CHARS:
        source = _condense(source)

    system_prompt = (
        "You are an expert meeting analyst. Analyze the meeting transcript and return ONLY "
        "a valid JSON object (no markdown fences, no extra text) with exactly these keys:\n"
        '"title": short professional meeting title, max 8 words,\n'
        '"summary": professional meeting summary in bullet points,\n'
        '"action_items": numbered list, each with task, owner and deadline '
        "(write 'Not specified' if unknown); if none write 'No action items found.',\n"
        '"key_decisions": numbered list; if none write \'No key decisions found.\',\n'
        '"open_questions": numbered list of unresolved questions; '
        "if none write 'No open questions found.'.\n"
        "All values must be plain strings."
    )

    raw = call_llm([("system", system_prompt), ("human", source)], temperature=0.2)
    data = _parse_json(raw)

    if data is None:
        data = {
            "title": "Meeting Summary",
            "summary": raw,
            "action_items": "Not available.",
            "key_decisions": "Not available.",
            "open_questions": "Not available.",
        }

    result = {}
    for key in ["title", "summary", "action_items", "key_decisions", "open_questions"]:
        value = data.get(key, "-")
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(v) for v in value)
        result[key] = str(value).strip()
    return result