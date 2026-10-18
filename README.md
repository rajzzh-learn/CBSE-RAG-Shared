# CBSE-RAG-Shared

Shared Python package used by all four CBSE Class 12 RAG Tutor projects:

| Project | GitHub Repo |
|---------|-------------|
| Chemistry | `rajzzh-learn/Chemistry-RAG-Project` |
| Computer Science | `rajzzh-learn/CS-RAG-Project` |
| Mathematics | `rajzzh-learn/Maths-RAG-Project` |
| Physics | `rajzzh-learn/Physics-RAG-Project` |

## What is shared

| Module | What it provides |
|--------|-----------------|
| `shared_tutor/config.py` | `get_config()`, LLM defaults, Groq model aliases |
| `shared_tutor/rag_chain.py` | `build_rag_chain()`, `build_llm()`, `convert_history()`, `invoke_with_retry()` |
| `shared_tutor/validator.py` | Layer 3 — `validate_answer()` via Gemini 2.0 Flash Lite (free) |
| `shared_tutor/ingest.py` | `build_vector_store()`, `load_pdfs()` |
| `shared_tutor/file_utils.py` | `extract_text()`, `build_attachment()`, `image_to_base64_uri()` |

## 3-Layer Intelligence Stack

```
Layer 1 — Retrieve   ChromaDB MMR search on subject PDFs
Layer 2 — Generate   Groq (fast, free) answers using retrieved context
Layer 3 — Validate   Gemini 2.0 Flash Lite fact-checks the answer (free)
```

Layer 3 is **opt-in**. Add to `.env` or Streamlit Secrets:
```
ENABLE_VALIDATION=true
GOOGLE_API_KEY=your-key-here
```
Get a free Gemini key at https://aistudio.google.com/app/apikey

## Installing in a subject project

### Local development
```bash
pip install -e "/Users/rajeshkumarasamy/Workspace/CBSE-RAG-Shared"
```

### Streamlit Cloud (add to subject requirements.txt)
```
git+https://github.com/rajzzh-learn/CBSE-RAG-Shared.git
```

## Architecture

```
/Workspace/
├── CBSE-RAG-Shared/          ← this repo (one fix = all 4 projects updated)
│   └── shared_tutor/
├── Chemistry RAG Project/    ← own GitHub repo, own Streamlit Cloud URL
├── CS RAG Project/           ← own GitHub repo, own Streamlit Cloud URL
├── Maths RAG Project/        ← own GitHub repo, own Streamlit Cloud URL
└── Physics RAG Project/      ← own GitHub repo, own Streamlit Cloud URL
```
