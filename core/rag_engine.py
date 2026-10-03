from core.llm import call_llm
from core.vector_store import build_vector_store, load_vector_store, get_retriever

RAG_SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
"""


class RagChain:
    """app.py ke liye simple chain: .invoke(question) -> answer text."""

    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        return call_llm(
            [
                ("system", RAG_SYSTEM_PROMPT + context),
                ("human", question),
            ],
            temperature=0.3,
        )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=4)
    return RagChain(retriever)


def load_rag_chain():
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    return RagChain(retriever)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer :{answer}")
    return answer