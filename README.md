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
| `shared_tutor/rag_chain.py` | `build_rag_chain()`, `build_llm()`, `generate_full_answer()`, `convert_history()`, `invoke_with_retry()` |
| `shared_tutor/validator.py` | Layer 3 — `validate_answer()` via Gemini 2.0 Flash Lite (free) |
| `shared_tutor/ingest.py` | `build_vector_store()`, `load_pdfs()` |
| `shared_tutor/file_utils.py` | `extract_text()`, `build_attachment()`, `image_to_base64_uri()` |

## 3-Layer Intelligence Stack

```
Layer 1 — Retrieve   ChromaDB MMR search on subject PDFs
Layer 2 — Generate   Groq (fast, free) answers using retrieved context
Layer 3 — Validate   Gemini 2.0 Flash Lite fact-checks the answer (free)
```

Layer 3 is **opt-in** and multi-LLM: Gemini and OpenAI GPT-4o validate in turn (`VALIDATION_PROVIDER` = first choice, the other is the fallback; each needs its own key). Long answers are validated section by section. When an answer is not verified the app shows why. Add to `.env` or Streamlit Secrets:
```
ENABLE_VALIDATION=true
GOOGLE_API_KEY=your-key-here
```
Get a free Gemini key at https://aistudio.google.com/app/apikey

Answers are checked in sections of about 5000 characters, so long answers are fully validated; answers over about 40000 characters are skipped, with the reason shown.

### Long answers
`generate_full_answer()` detects when the model stopped because it hit the token limit
(`finish_reason=length`) and asks it to continue, up to 4 extra calls, joining the parts.
Long outputs (for example many years of PYQs with solutions) are therefore not cut off.

## Installing in a subject project

### Local development
```bash
pip install -e "/Users/rajeshkumarasamy/Workspace/CBSE-RAG-Shared"
```

### Streamlit Cloud (add to subject requirements.txt, pinned to a release tag)
```
shared-tutor @ git+https://github.com/rajzzh-learn/CBSE-RAG-Shared.git@v1.0.1
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
