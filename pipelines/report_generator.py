"""
Project Consilium — LLM Report Generator
==========================================
Converts structured pipeline findings into a formatted
clinical radiology report using a local Ollama LLM.

Fixes:
  - Ollama list() returns a ListResponse object, not a dict
  - Section parser now handles phi3's inline-header style:
      "Findings: text..." (not just "Findings:\ntext...")
  - Handles bold markdown headers: **Findings:**
  - Fallback model preference list includes phi3:mini and phi3:latest
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Preferred models — first available is used
# ---------------------------------------------------------------------------

PREFERRED_MODELS = [
    "phi3:mini",
    "phi3:latest",
    "phi3",
    "phi3:3.8b",
    "llama3.2:3b",
    "llama3.2",
    "mistral",
    "gemma2:2b",
    "gemma:2b",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ReportResult:
    findings: str
    impression: str
    recommendation: str
    model_used: str
    raw_response: str

    @property
    def full_report(self) -> str:
        return (
            f"**Findings:**\n{self.findings}\n\n"
            f"**Impression:**\n{self.impression}\n\n"
            f"**Recommendation:**\n{self.recommendation}"
        )


# ---------------------------------------------------------------------------
# Ollama model discovery
# ---------------------------------------------------------------------------

def get_available_ollama_models() -> list[str]:
    """
    Return list of model names currently pulled in Ollama.
    Returns empty list if Ollama is not reachable.

    The ollama Python SDK returns a ListResponse object (not a dict),
    so we access .models as an attribute and handle both old and new
    SDK versions defensively.
    """
    try:
        import ollama
        response = ollama.list()

        # New SDK: ListResponse with .models attribute
        models_raw = getattr(response, "models", None)
        if models_raw is not None:
            names = []
            for m in models_raw:
                name = (
                    getattr(m, "model", None)
                    or getattr(m, "name",  None)
                    or (m.get("model") if isinstance(m, dict) else None)
                    or (m.get("name")  if isinstance(m, dict) else None)
                )
                if name:
                    names.append(name)
            return names

        # Old SDK fallback: response is a plain dict
        if isinstance(response, dict):
            return [
                m.get("model") or m.get("name", "")
                for m in response.get("models", [])
                if m.get("model") or m.get("name")
            ]

        return []

    except Exception:
        return []


def select_model(preferred: list[str] = PREFERRED_MODELS) -> Optional[str]:
    """
    Return the first preferred model that is available in Ollama.
    Returns None if Ollama is unreachable or no preferred model is pulled.
    """
    available = get_available_ollama_models()
    if not available:
        return None

    available_full = set(available)

    for candidate in preferred:
        # Exact match
        if candidate in available_full:
            return candidate

        # Base name match (e.g. "phi3" matches "phi3:latest")
        base = candidate.split(":")[0]
        for a in available:
            if a.split(":")[0] == base:
                return a

    # Nothing preferred found — return whatever is available
    return available[0]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "System: You are an expert radiologist. Synthesize the provided AI findings into a concise clinical report. "
    "Do not repeat the exact numerical voxel counts or percentages in the Impression section; "
    "reserve the Impression strictly for clinical synthesis, diagnosis, and next steps."
)


def build_report_prompt(
    scan_type: str,
    findings_text: str,
    extra_context: str = "",
) -> str:
    context_block = f"\nAdditional context: {extra_context}\n" if extra_context else ""

    return f"""Scan type: {scan_type}
{context_block}
AI-detected findings:
{findings_text}

Write a radiology report using EXACTLY this format.
Each section label must be on its own line followed by the content.
Do not use markdown bold (**) or any other formatting.

Findings:
[One paragraph, max 60 words]

Impression:
[One or two sentences, max 30 words]

Recommendation:
[One or two actionable recommendations, max 30 words]

Rules:
- Do NOT invent patients, history, or demographics.
- Do NOT go beyond the findings listed above.
- Use cautious, hedged medical language.
- Stop immediately after Recommendation. No extra text.""".strip()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_report(
    scan_type: str,
    findings_text: str,
    extra_context: str = "",
    temperature: float = 0.15,
    max_tokens: int = 250,
    model_override: Optional[str] = None,
) -> ReportResult:
    """
    Generate a structured radiology report via Ollama.

    Raises RuntimeError if Ollama is unavailable or no model found.
    """
    import ollama

    model = model_override or select_model()
    if model is None:
        available = get_available_ollama_models()
        if available:
            model = available[0]
        else:
            raise RuntimeError(
                "Ollama is not running or no models are available.\n"
                "Start Ollama and pull a model:\n"
                "  ollama pull phi3:mini"
            )

    prompt = build_report_prompt(scan_type, findings_text, extra_context)

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop": ["---", "Note:", "Example:", "Teaching"],
        },
    )

    # Handle both dict-style and object-style response from ollama SDK
    msg = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
    if isinstance(msg, dict):
        raw = msg.get("content", "").strip()
    else:
        raw = getattr(msg, "content", "").strip()

    findings, impression, recommendation = _parse_report_sections(raw)

    return ReportResult(
        findings=findings,
        impression=impression,
        recommendation=recommendation,
        model_used=model,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Section parser
# ---------------------------------------------------------------------------

def _parse_report_sections(raw: str) -> tuple[str, str, str]:
    """
    Extract the three sections from the LLM output.

    Handles phi3's actual output styles:
      Style A -- label on own line:   Findings:\n Some text...
      Style B -- label inline:        Findings: Some text on same line...
      Style C -- bold markdown:       **Findings:** Some text...

    The old parser only handled Style A, causing Impression and
    Recommendation to show as "Not generated." when phi3 used Style B.
    """
    sections: dict[str, str] = {
        "findings":       "",
        "impression":     "",
        "recommendation": "",
    }
    current: Optional[str] = None

    # Strip markdown bold markers (**text** -> text)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", raw)

    HEADER_MAP = {
        "findings":        "findings",
        "finding":         "findings",
        "impression":      "impression",
        "impressions":     "impression",
        "recommendation":  "recommendation",
        "recommendations": "recommendation",
        "recommend":       "recommendation",
    }

    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        matched_section: Optional[str] = None
        matched_rest:    str           = ""

        for keyword, section_name in HEADER_MAP.items():
            # Match keyword at start of line, with optional colon and spaces
            pattern = re.compile(rf"^{re.escape(keyword)}\s*:?\s*", re.IGNORECASE)
            m = pattern.match(stripped)
            if m:
                matched_section = section_name
                matched_rest    = stripped[m.end():].strip()
                break

        if matched_section:
            current = matched_section
            if matched_rest:
                sections[current] += matched_rest + " "
            continue

        if current:
            sections[current] += stripped + " "

    # Trim whitespace from all sections
    for key in sections:
        sections[key] = sections[key].strip()

    # Fallback: if nothing was parsed, put everything in findings
    if not any(sections.values()):
        sections["findings"]       = raw.strip()
        sections["impression"]     = "See findings above."
        sections["recommendation"] = "Clinical correlation recommended."

    return (
        sections["findings"]       or "Not generated.",
        sections["impression"]     or "Not generated.",
        sections["recommendation"] or "Clinical correlation recommended.",
    )


# ---------------------------------------------------------------------------
# Ollama health check (used by sidebar UI)
# ---------------------------------------------------------------------------

def get_ollama_status() -> dict:
    """
    Return a status dict for display in the Streamlit sidebar.
    """
    models   = get_available_ollama_models()
    selected = select_model() if models else None
    reachable = bool(models)

    if not reachable:
        message = (
            "Ollama is not running. "
            "It may already be running as a background service on Windows. "
            "Check with: ollama list"
        )
    elif not selected:
        model_list = ", ".join(PREFERRED_MODELS[:4])
        message = (
            f"No preferred model found. Pull one of: {model_list}. "
            f"Available: {', '.join(models)}."
        )
    else:
        message = f"Ready — using {selected}."

    return {
        "reachable": reachable,
        "models":    models,
        "selected":  selected,
        "message":   message,
    }
