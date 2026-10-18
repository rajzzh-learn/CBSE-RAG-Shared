"""
Shared configuration helpers for all CBSE RAG Tutor projects.

Resolution order for every config key:
  1. st.secrets   — Streamlit Community Cloud secrets
  2. os.environ   — local .env / shell exports (via python-dotenv)
  3. default      — hard-coded fallback

Subject-specific paths (PDF_DIRS, VECTOR_STORE_DIR) are NOT defined here;
they live in each subject's own config.py because they point to that repo's
folder structure.  Only the keys that are truly identical across all subjects
are centralised here.
"""
from __future__ import annotations
import os


def get_config(key: str, default: str = "") -> str:
    """Resolve a config value: st.secrets → os.environ → default."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


# ── LLM defaults ───────────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER   = "groq"
DEFAULT_GROQ_MODEL     = "openai/gpt-oss-120b"
DEFAULT_OPENAI_MODEL   = "gpt-4o"
DEFAULT_WATSONX_MODEL  = "ibm/granite-3-3-8b-instruct"
DEFAULT_WATSONX_URL    = "https://us-south.ml.cloud.ibm.com"

# ── Embedding ──────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────────
DEFAULT_RETRIEVER_TOP_K = 6
DEFAULT_CHUNK_SIZE      = 800
DEFAULT_CHUNK_OVERLAP   = 120

# ── Groq model aliases (keep in sync when Groq renames models) ─────────────
GROQ_MODEL_ALIASES: dict[str, str] = {
    "llama-3.3-70b-versatile": "openai/gpt-oss-120b",
    "llama-3.1-8b-instant":    "openai/gpt-oss-20b",
    "llama3-70b-8192":         "openai/gpt-oss-120b",
    "llama3-8b-8192":          "openai/gpt-oss-20b",
    "mixtral-8x7b-32768":      "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b":        "openai/gpt-oss-120b",
}

# ── Validation (Layer 3) ───────────────────────────────────────────────────
# Set ENABLE_VALIDATION=true in .env / Streamlit Secrets to activate.
# VALIDATION_PROVIDER: "gemini" (free, default) or "openai" (GPT-4o, paid).
DEFAULT_VALIDATION_PROVIDER = "gemini"
DEFAULT_GEMINI_VALIDATOR_MODEL = "gemini-2.0-flash-lite"
