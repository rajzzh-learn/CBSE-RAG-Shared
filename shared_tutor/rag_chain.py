"""
Shared RAG chain helpers for all CBSE Tutor projects.

Each subject imports from here instead of maintaining its own copy:

    from shared_tutor.rag_chain import (
        build_rag_chain,
        convert_history,
        invoke_with_retry,
        is_rate_limit_error,
    )

build_rag_chain() takes two subject-specific arguments:
  - vector_store  : the subject's ChromaDB instance
  - system_prompt : the subject's SYSTEM_PROMPT string (contains {context})

Everything else (LLM selection, retry logic, history compaction) is shared.
"""
from __future__ import annotations

import re
import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from shared_tutor.config import (
    DEFAULT_GEMINI_VALIDATOR_MODEL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_WATSONX_MODEL,
    DEFAULT_WATSONX_URL,
    DEFAULT_VALIDATION_PROVIDER,
    GROQ_MODEL_ALIASES,
    get_config,
)


# ── Rate-limit helpers ─────────────────────────────────────────────────────

def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("rate_limit", "rate limit", "429", "quota",
                                  "tokens per minute", "tpm"))


def _retry_delay(exc: Exception, default: float) -> float:
    m = re.search(r"try again in\s+(?:(\d+)m)?\s*([\d.]+)s", str(exc), re.IGNORECASE)
    if m:
        return min(float(m.group(1) or 0) * 60 + float(m.group(2)) + 1.0, 30.0)
    return default


def invoke_with_retry(fn, attempts: int = 5):
    """Call fn(); on rate-limit errors wait for the advertised delay and retry."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_rate_limit_error(exc):
                raise
            delay = _retry_delay(exc, default=min(2.5 * (attempt + 1), 15.0))
            time.sleep(delay)


# ── LLM factory ───────────────────────────────────────────────────────────

def build_llm():
    """Instantiate the LLM based on LLM_PROVIDER config. Called by each subject."""
    provider = get_config("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).lower()

    if provider == "watsonx":
        from langchain_ibm import WatsonxLLM
        apikey = get_config("WATSONX_APIKEY") or get_config("WATSONX_API_KEY")
        return WatsonxLLM(
            model_id=get_config("WATSONX_MODEL_ID") or get_config("WATSONX_MODEL", DEFAULT_WATSONX_MODEL),
            url=get_config("WATSONX_URL", DEFAULT_WATSONX_URL),
            apikey=apikey,
            project_id=get_config("WATSONX_PROJECT_ID"),
            params={"max_new_tokens": 8192, "temperature": 0.3, "repetition_penalty": 1.1},
        )

    elif provider == "groq":
        from langchain_openai import ChatOpenAI
        api_key = get_config("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing `GROQ_API_KEY`. Add it to your .env file or Streamlit Secrets."
            )
        raw_model = get_config("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        model = GROQ_MODEL_ALIASES.get(raw_model, raw_model)

        primary = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=8192,
            max_retries=3,
        )
        fallback = ChatOpenAI(
            model="openai/gpt-oss-20b",
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=8192,
            max_retries=3,
        )
        return primary.with_fallbacks([fallback])

    else:  # openai
        from langchain_openai import ChatOpenAI
        api_key = get_config("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing `OPENAI_API_KEY`. Add it to your .env file or Streamlit Secrets."
            )
        return ChatOpenAI(
            model=get_config("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            api_key=api_key,
            temperature=0.3,
            max_tokens=8192,
        )


# ── RAG chain builder ─────────────────────────────────────────────────────

def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


MAX_CONTINUATIONS = 4  # extra LLM calls allowed when an answer hits max_tokens
_CONTINUE_PROMPT = (
    "Your previous reply was cut off by the output limit. Continue EXACTLY from where "
    "it stopped. Do not repeat anything already written, do not add a preamble or "
    "summary, and keep the same formatting (finish any open LaTeX, table or list)."
)


def _was_truncated(msg) -> bool:
    meta = getattr(msg, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("stop_reason") or ""
    return str(reason).lower() in ("length", "max_tokens", "max_new_tokens", "token_limit")


def generate_full_answer(llm, prompt, inputs: dict) -> str:
    """Invoke llm and keep asking it to continue while it stops on the token limit.

    Without this, long answers (e.g. 10+ years of PYQs with solutions) are silently
    cut off at max_tokens.
    """
    messages = prompt.format_messages(**inputs)
    first = invoke_with_retry(lambda: llm.invoke(messages))
    text = first if isinstance(first, str) else (first.content or "")
    last = first
    for _ in range(MAX_CONTINUATIONS):
        if isinstance(last, str) or not _was_truncated(last):
            break
        follow_up = messages + [AIMessage(content=text), HumanMessage(content=_CONTINUE_PROMPT)]
        last = invoke_with_retry(lambda: llm.invoke(follow_up))
        text += last.content or ""
    return text


def build_rag_chain(vector_store, system_prompt: str, retriever_top_k: int = 6):
    """
    Build a subject-specific RAG chain.

    Args:
        vector_store:    The subject's ChromaDB Chroma instance.
        system_prompt:   The subject's SYSTEM_PROMPT string (must contain {context}).
        retriever_top_k: Number of chunks to retrieve (default 6).

    Returns a chain: chain.invoke({"question": str, "chat_history": list})
    → {"answer": str, "source_documents": list}
    """
    llm = build_llm()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": retriever_top_k, "fetch_k": max(30, retriever_top_k * 3)},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Retrieve once → reuse for both context text and source_documents
    chain = (
        RunnablePassthrough.assign(
            source_documents=RunnableLambda(lambda x: retriever.invoke(x["question"]))
        )
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: _format_docs(x["source_documents"]))
        )
        | RunnablePassthrough.assign(
            answer=RunnableLambda(lambda x: generate_full_answer(llm, prompt, x))
        )
    )
    return chain


# ── Smart history compaction ───────────────────────────────────────────────

def convert_history(
    raw_history: list[dict],
    *,
    token_budget: int = 12000,      # max total chars of history sent to the LLM
    recent_full_turns: int = 6,     # most-recent N turns kept verbatim
    max_user_chars: int = 2000,     # cap per user message in recent turns
    max_assistant_chars: int = 6000,# cap per assistant message in recent turns
    summary_chars: int = 300,       # chars kept from older turns
) -> list:
    """Convert chat history to LangChain messages with smart token-budget management.

    Strategy:
    - The most recent `recent_full_turns` user+assistant pairs are kept verbatim
      (capped at max_user_chars / max_assistant_chars).
    - Older turns are compacted to `summary_chars` so the model retains topic
      awareness without blowing the context window.
    - Hard ceiling `token_budget` ensures safety at 20+ iteration sessions.
    """
    # Group flat message list into (user, assistant|None) turn pairs
    turns: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(raw_history):
        msg = raw_history[i]
        if msg["role"] == "user":
            nxt = raw_history[i + 1] if i + 1 < len(raw_history) else None
            if nxt and nxt["role"] == "assistant":
                turns.append((msg, nxt))
                i += 2
            else:
                turns.append((msg, None))
                i += 1
        else:
            i += 1  # stray assistant message — skip

    messages: list = []
    total_chars = 0
    cutoff = max(0, len(turns) - recent_full_turns)

    for idx, (u_msg, a_msg) in enumerate(turns):
        in_recent = idx >= cutoff

        # User message
        u_text = u_msg.get("content", "")
        u_text = u_text[:max_user_chars] if in_recent else u_text[:summary_chars]
        if total_chars + len(u_text) > token_budget:
            break
        messages.append(HumanMessage(content=u_text))
        total_chars += len(u_text)

        # Assistant message
        if a_msg is None:
            continue
        a_text = a_msg.get("content", "")
        if in_recent:
            a_text = a_text[:max_assistant_chars]
        else:
            a_text = a_text[:summary_chars].rstrip() + " …"

        if total_chars + len(a_text) > token_budget:
            messages.append(AIMessage(content=a_text[:summary_chars].rstrip() + " …"))
            break
        messages.append(AIMessage(content=a_text))
        total_chars += len(a_text)

    return messages
