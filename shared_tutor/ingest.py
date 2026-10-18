"""
Subject-agnostic PDF ingestion pipeline.

Each subject calls build_vector_store() passing its own pdf_dirs,
vector_store_dir, source_labels, and embedding_model — all of which
come from that subject's config.py.  This file contains zero
subject-specific knowledge.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Force CPU embeddings on Streamlit Cloud (no MPS / CUDA available)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/st_cache")


def load_pdfs(pdf_dirs: list[Path], source_labels: dict | None = None) -> list:
    """Load every PDF found under the given directories.

    Args:
        pdf_dirs:      List of directories to scan for *.pdf files.
        source_labels: Optional dict mapping Path → human-readable label string.
                       Matched documents receive a ``source_label`` metadata key.
    """
    from langchain_community.document_loaders import PyPDFLoader

    source_labels = source_labels or {}
    docs = []
    for directory in pdf_dirs:
        if not directory.exists():
            logger.warning("Directory not found, skipping: %s", directory)
            continue
        label = source_labels.get(directory)
        for pdf_path in sorted(directory.glob("*.pdf")):
            logger.info("Loading: %s", pdf_path.name)
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                if label:
                    for doc in pages:
                        doc.metadata["source_label"] = label
                docs.extend(pages)
            except Exception as exc:
                logger.warning("Could not load %s: %s", pdf_path.name, exc)
    logger.info("Total pages loaded: %d", len(docs))
    return docs


def build_vector_store(
    pdf_dirs: list[Path],
    vector_store_dir: Path,
    embedding_model: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    source_labels: dict | None = None,
    force_rebuild: bool = False,
):
    """Build (or load) a ChromaDB vector store for a subject.

    Args:
        pdf_dirs:          Directories containing the subject's PDFs.
        vector_store_dir:  Where to persist the ChromaDB data.
        embedding_model:   HuggingFace model name for embeddings.
        chunk_size:        Text splitter chunk size (chars).
        chunk_overlap:     Text splitter overlap (chars).
        source_labels:     Optional display-name mapping for source dirs.
        force_rebuild:     If True, delete and re-ingest everything.

    Returns:
        A langchain_chroma.Chroma instance ready for retrieval.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if vector_store_dir.exists() and not force_rebuild:
        logger.info("Loading existing vector store from %s", vector_store_dir)
        return Chroma(
            persist_directory=str(vector_store_dir),
            embedding_function=embeddings,
        )

    logger.info("Building vector store — this may take a few minutes …")
    raw_docs = load_pdfs(pdf_dirs, source_labels)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    logger.info("Total chunks: %d", len(chunks))

    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir, ignore_errors=True)
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vector_store_dir),
    )
    logger.info("Vector store saved to %s", vector_store_dir)
    return vector_store
