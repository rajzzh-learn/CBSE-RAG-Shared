"""
shared_tutor — shared code for all CBSE RAG Tutor projects.

Exports the key public symbols so subjects can do:
    from shared_tutor import get_config, build_rag_chain, convert_history,
                             validate_answer, build_vector_store, extract_text
"""
from shared_tutor.config import get_config
from shared_tutor.rag_chain import (
    build_rag_chain,
    build_llm,
    convert_history,
    invoke_with_retry,
    is_rate_limit_error,
)
from shared_tutor.validator import validate_answer, ValidationResult
from shared_tutor.ingest import build_vector_store, load_pdfs
from shared_tutor.file_utils import (
    extract_text,
    build_attachment,
    image_to_base64_uri,
    is_image,
    Attachment,
    SUPPORTED_EXTS,
    SUPPORTED_IMAGE_EXTS,
    TOTAL_TEXT_BUDGET,
)

__all__ = [
    "get_config",
    "build_rag_chain",
    "build_llm",
    "convert_history",
    "invoke_with_retry",
    "is_rate_limit_error",
    "validate_answer",
    "ValidationResult",
    "build_vector_store",
    "load_pdfs",
    "extract_text",
    "build_attachment",
    "image_to_base64_uri",
    "is_image",
    "Attachment",
    "SUPPORTED_EXTS",
    "SUPPORTED_IMAGE_EXTS",
    "TOTAL_TEXT_BUDGET",
]
