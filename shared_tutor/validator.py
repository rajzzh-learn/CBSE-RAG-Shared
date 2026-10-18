"""
Layer 3 — multi-LLM answer validator.

After the tutor LLM (Groq / OpenAI / watsonx) produces an answer, a *different* LLM
fact-checks it against the retrieved study material. Validators are tried in order and
the next one is used if a provider fails, so a verified tick does not depend on a single
vendor being up.

Supported validators (all optional, enabled by their API key):
    gemini  -> GOOGLE_API_KEY   (model: GEMINI_VALIDATOR_MODEL, default gemini-2.0-flash-lite)
    openai  -> OPENAI_API_KEY   (model: OPENAI_VALIDATOR_MODEL, default gpt-4o)

Configuration:
    ENABLE_VALIDATION=true
    VALIDATION_PROVIDER=gemini            # first choice; the other provider is the fallback
                                          # (comma list allowed, e.g. "openai,gemini")

Long answers are validated section by section so nothing is clipped.

validate_answer() never raises. The result always carries a `status` and, when the answer
was not verified, a human-readable `reason`:
    verified  - every section confirmed correct
    corrected - at least one section was fixed (see `corrections`)
    skipped   - validation did not run (disabled, no key, too long) - see `reason`
    error     - every validator failed - see `reason`
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from shared_tutor.config import (
    DEFAULT_GEMINI_VALIDATOR_MODEL,
    get_config,
)

DEFAULT_OPENAI_VALIDATOR_MODEL = "gpt-4o"
ALL_PROVIDERS = ("gemini", "openai")

MAX_CHUNK_CHARS = 5000      # size of each section sent to a validator
MAX_CHUNKS = 8              # more than this -> skipped (cost / rate-limit guard)
MAX_CONTEXT_CHARS = 12000   # retrieved material shown to the validator
MIN_CORRECTED_RATIO = 0.5   # a "corrected" section far shorter than the original is rejected


@dataclass
class ValidationResult:
    answer: str                      # final answer to show the student
    was_corrected: bool = False      # True  -> a validator found and fixed errors
    is_verified: bool = False        # True  -> validators confirmed the answer is correct
    corrections: list[str] = field(default_factory=list)
    status: str = "skipped"          # "verified" | "corrected" | "skipped" | "error"
    provider: str = ""               # validator(s) that produced the result
    reason: str = ""                 # why the answer is not verified (skipped / error)


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

A tutor gave this answer{part_note}:
--- TUTOR ANSWER ---
{answer}

Your task:
1. Check every fact, formula, reaction, equation, value, and mark allocation in the \
   tutor answer against the source material and your own expert knowledge.
2. If the tutor lists past-year / board questions, every such question must be present in \
   the SOURCE MATERIAL. A question that is not in the source material, or that repeats \
   another question already listed, is an error: remove it and say so in the corrections.
3. Judge only the text shown. Do not complain that other parts of a long answer are missing.

If the answer is fully correct, reply with exactly:
STATUS: CORRECT
VERIFIED ANSWER:
<repeat the tutor answer verbatim, no changes>

If anything is wrong or missing, reply with:
STATUS: CORRECTED
CORRECTIONS MADE:
- <bullet 1: describe what was wrong and what the correct version is>
- <bullet 2: ...>
CORRECTED ANSWER:
<the complete corrected text, ready to show to the student>

Do NOT add unnecessary caveats. Do NOT change correct parts. Be concise and accurate.\
"""


# ── Provider back-ends ─────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=get_config("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=get_config("GEMINI_VALIDATOR_MODEL", DEFAULT_GEMINI_VALIDATOR_MODEL),
        contents=prompt,
    )
    return response.text or ""


def _call_openai(prompt: str) -> str:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=get_config("OPENAI_VALIDATOR_MODEL", DEFAULT_OPENAI_VALIDATOR_MODEL),
        api_key=get_config("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=8192,
        timeout=90,
        max_retries=2,
    )
    return llm.invoke(prompt).content or ""


# name -> (api-key config name, callable). Kept in a dict so tests can swap callables.
_BACKENDS = {
    "gemini": ("GOOGLE_API_KEY", _call_gemini),
    "openai": ("OPENAI_API_KEY", _call_openai),
}


def _provider_order() -> list[str]:
    configured = [p.strip().lower() for p in get_config("VALIDATION_PROVIDER", "gemini").split(",")]
    order = [p for p in configured if p in _BACKENDS]
    order += [p for p in ALL_PROVIDERS if p not in order]   # remaining ones are fallbacks
    return order


def _available_providers() -> list[str]:
    return [p for p in _provider_order() if get_config(_BACKENDS[p][0])]


# ── Main entry point ───────────────────────────────────────────────────────

def validate_answer(subject: str, question: str, context: str, answer: str) -> ValidationResult:
    """Validate `answer` against `context`. Always safe to call."""
    if get_config("ENABLE_VALIDATION", "false").lower() != "true":
        return ValidationResult(answer=answer, status="skipped",
                                reason="Validation is switched off (ENABLE_VALIDATION is not 'true').")

    providers = _available_providers()
    if not providers:
        return ValidationResult(
            answer=answer, status="skipped",
            reason="No validator key configured (set GOOGLE_API_KEY and/or OPENAI_API_KEY).")

    chunks = _split_answer(answer, MAX_CHUNK_CHARS)
    if len(chunks) > MAX_CHUNKS:
        return ValidationResult(
            answer=answer, status="skipped",
            reason=f"Answer is too long to validate ({len(answer)} characters).")

    out_parts: list[str] = []
    corrections: list[str] = []
    used: list[str] = []
    any_corrected = False
    failures: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        part_note = f" (section {idx} of {len(chunks)})" if len(chunks) > 1 else ""
        prompt = _VALIDATOR_PROMPT.format(
            subject=subject,
            question=question[:3000],
            context=context[:MAX_CONTEXT_CHARS],
            answer=chunk,
            part_note=part_note,
        )
        outcome = _validate_chunk(prompt, chunk, providers, failures)
        if outcome is None:
            return ValidationResult(
                answer=answer, status="error",
                reason="Every validator failed: " + "; ".join(failures[-len(providers):]))
        status, text, notes, provider = outcome
        out_parts.append(text)
        used.append(provider)
        if status == "corrected":
            any_corrected = True
            corrections.extend(notes)

    final = "\n\n".join(out_parts) if len(chunks) > 1 else out_parts[0]
    provider_label = " + ".join(dict.fromkeys(used))
    if any_corrected:
        return ValidationResult(answer=final, was_corrected=True, corrections=corrections,
                                status="corrected", provider=provider_label)
    return ValidationResult(answer=answer, is_verified=True, status="verified",
                            provider=provider_label)


def _validate_chunk(prompt: str, chunk: str, providers: list[str], failures: list[str]):
    """Try each validator in turn. Returns (status, text, corrections, provider) or None."""
    for name in providers:
        try:
            raw = _BACKENDS[name][1](prompt)
            parsed = _parse_validator_response(raw, original_answer=chunk)
        except Exception as exc:  # never crash the student-facing app
            failures.append(f"{name}: {type(exc).__name__}")
            continue
        if parsed is None:
            failures.append(f"{name}: unreadable reply")
            continue
        status, text, notes = parsed
        if status == "corrected" and len(text) < MIN_CORRECTED_RATIO * len(chunk):
            failures.append(f"{name}: corrected text was cut short")
            continue
        return status, text, notes, name
    return None


# ── Response parser ────────────────────────────────────────────────────────

def _parse_validator_response(raw: str, original_answer: str):
    """Return (status, text, corrections) or None if the reply cannot be understood.

    CORRECTED is checked first: "STATUS: CORRECT" is a prefix of "STATUS: CORRECTED".
    """
    raw = raw.strip()

    if re.search(r"STATUS:\s*CORRECTED\b", raw, re.IGNORECASE):
        corrections: list[str] = []
        block = re.search(r"CORRECTIONS MADE:\s*\n(.*?)(?:CORRECTED ANSWER:|$)",
                          raw, re.IGNORECASE | re.DOTALL)
        if block:
            for line in block.group(1).splitlines():
                line = line.strip().lstrip("-• ")
                if line:
                    corrections.append(line)
        m = re.search(r"CORRECTED ANSWER:\s*\n?(.*)", raw, re.IGNORECASE | re.DOTALL)
        if not m or not m.group(1).strip():
            return None
        return "corrected", m.group(1).strip(), corrections

    if re.search(r"STATUS:\s*CORRECT\b", raw, re.IGNORECASE):
        # Always keep the tutor's own text: a validator may silently shorten what it echoes.
        return "verified", original_answer, []

    return None


# ── Helpers ────────────────────────────────────────────────────────────────

def _split_answer(text: str, limit: int) -> list[str]:
    """Split on blank lines into pieces of at most ~`limit` characters."""
    if len(text) <= limit:
        return [text]
    pieces: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        while len(para) > limit:                       # one huge paragraph
            if current:
                pieces.append(current)
                current = ""
            pieces.append(para[:limit])
            para = para[limit:]
        if current and len(current) + len(para) + 2 > limit:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def show_validation_badge(validation: ValidationResult) -> None:
    """Render the validation outcome in Streamlit. One call replaces per-app if/elif code."""
    import streamlit as st

    if validation.status == "verified":
        st.success(f"✅ Answer verified by {validation.provider}")
    elif validation.status == "corrected":
        st.warning(f"⚠️ **Answer corrected by {validation.provider}**")
        for note in validation.corrections:
            st.caption(f"• {note}")
    elif validation.reason:
        st.caption(f"ℹ️ Not verified — {validation.reason}")
