"""
Layer 3 — Answer Validator (Gemini 2.0 Flash Lite, free tier).

Usage in any subject app.py:

    from shared_tutor.validator import validate_answer, ValidationResult

    result = validate_answer(
        subject="Chemistry",
        question=user_input,
        context=retrieved_context_text,
        answer=primary_answer,
    )
    if result.was_corrected:
        # show result.answer with a ⚠️ Corrected badge
    else:
        # show result.answer with a ✅ Verified badge

Set ENABLE_VALIDATION=true in .env / Streamlit Secrets to activate.
If GOOGLE_API_KEY is absent or ENABLE_VALIDATION != "true", validate_answer()
returns the original answer unchanged with status="skipped".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from shared_tutor.config import DEFAULT_GEMINI_VALIDATOR_MODEL, get_config


@dataclass
class ValidationResult:
    answer: str                      # final answer to show the student
    was_corrected: bool = False      # True  → Gemini found and fixed errors
    is_verified: bool  = False       # True  → Gemini confirmed answer is correct
    corrections: list[str] = field(default_factory=list)  # list of corrections made
    status: str = "skipped"          # "verified" | "corrected" | "skipped" | "error"


# ── Validator prompt ───────────────────────────────────────────────────────

_VALIDATOR_PROMPT = """\
You are a strict CBSE Class 12 {subject} examiner and fact-checker.

A student asked the following question:
--- QUESTION ---
{question}

The following chunks were retrieved from the official study material (NCERT textbooks, \
exemplar, PYQs):
--- SOURCE MATERIAL ---
{context}

A tutor gave this answer:
--- TUTOR ANSWER ---
{answer}

Your task:
1. Check every fact, formula, reaction, equation, value, and mark allocation in the \
   tutor answer against the source material and your own expert knowledge.
2. If the answer is fully correct, reply with exactly:
   STATUS: CORRECT
   VERIFIED ANSWER:
   <repeat the tutor answer verbatim, no changes>

3. If anything is wrong or missing, reply with:
   STATUS: CORRECTED
   CORRECTIONS MADE:
   - <bullet 1: describe what was wrong and what the correct version is>
   - <bullet 2: ...>
   CORRECTED ANSWER:
   <the complete corrected answer, ready to show to the student>

Do NOT add unnecessary caveats. Do NOT change correct parts. Be concise and accurate.\
"""


# ── Main entry point ───────────────────────────────────────────────────────

def validate_answer(
    subject: str,
    question: str,
    context: str,
    answer: str,
) -> ValidationResult:
    """
    Validate `answer` against `context` using Gemini 2.0 Flash Lite (free).

    Returns a ValidationResult. Always safe to call — falls back gracefully
    when validation is disabled or GOOGLE_API_KEY is missing.
    """
    # ── Guard: is validation enabled? ──────────────────────────────────────
    if get_config("ENABLE_VALIDATION", "false").lower() != "true":
        return ValidationResult(answer=answer, status="skipped")

    google_key = get_config("GOOGLE_API_KEY")
    if not google_key:
        return ValidationResult(answer=answer, status="skipped")

    provider = get_config("VALIDATION_PROVIDER", "gemini").lower()
    if provider != "gemini":
        # openai path — future extension; skip for now
        return ValidationResult(answer=answer, status="skipped")

    # ── Call Gemini ─────────────────────────────────────────────────────────
    try:
        from google import genai

        prompt = _VALIDATOR_PROMPT.format(
            subject=subject,
            question=question[:3000],        # cap to avoid huge prompts
            context=context[:4000],
            answer=answer[:6000],
        )

        client = genai.Client(api_key=google_key)
        response = client.models.generate_content(
            model=get_config("GEMINI_VALIDATOR_MODEL", DEFAULT_GEMINI_VALIDATOR_MODEL),
            contents=prompt,
        )
        raw = response.text or ""

        return _parse_validator_response(raw, original_answer=answer)

    except Exception as exc:
        # Never crash the student-facing app because of a validator failure
        return ValidationResult(answer=answer, status="error",
                                corrections=[f"Validator error (ignored): {exc}"])


# ── Response parser ────────────────────────────────────────────────────────

def _parse_validator_response(raw: str, original_answer: str) -> ValidationResult:
    """Parse the structured Gemini validator response."""
    raw = raw.strip()

    # ── CORRECT path ───────────────────────────────────────────────────────
    if re.search(r"STATUS:\s*CORRECT", raw, re.IGNORECASE):
        # Extract everything after "VERIFIED ANSWER:"
        m = re.search(r"VERIFIED ANSWER:\s*\n?(.*)", raw, re.IGNORECASE | re.DOTALL)
        verified_text = m.group(1).strip() if m else original_answer
        # If Gemini just echoed the original, use the original to avoid subtle changes
        if _similarity(verified_text, original_answer) > 0.85:
            verified_text = original_answer
        return ValidationResult(
            answer=verified_text,
            is_verified=True,
            status="verified",
        )

    # ── CORRECTED path ─────────────────────────────────────────────────────
    if re.search(r"STATUS:\s*CORRECTED", raw, re.IGNORECASE):
        # Extract corrections list
        corrections: list[str] = []
        corr_block = re.search(
            r"CORRECTIONS MADE:\s*\n(.*?)(?:CORRECTED ANSWER:|$)",
            raw, re.IGNORECASE | re.DOTALL
        )
        if corr_block:
            for line in corr_block.group(1).splitlines():
                line = line.strip().lstrip("-• ")
                if line:
                    corrections.append(line)

        # Extract corrected answer
        m = re.search(r"CORRECTED ANSWER:\s*\n?(.*)", raw, re.IGNORECASE | re.DOTALL)
        corrected_text = m.group(1).strip() if m else original_answer

        return ValidationResult(
            answer=corrected_text,
            was_corrected=True,
            corrections=corrections,
            status="corrected",
        )

    # ── Fallback: could not parse — return original untouched ──────────────
    return ValidationResult(answer=original_answer, status="skipped")


def _similarity(a: str, b: str) -> float:
    """Very fast character-level Jaccard similarity (good enough for our use)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
