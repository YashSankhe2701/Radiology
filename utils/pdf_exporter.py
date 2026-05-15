"""
Project Consilium — PDF Report Exporter
=========================================
Generates a downloadable clinical PDF report from pipeline results.

Fix applied:
  Helvetica in fpdf2 only supports latin-1 (ISO 8859-1).
  Unicode characters like em-dash (—), bullet (•), smart quotes ("")
  must be replaced with ASCII equivalents BEFORE being passed to any
  pdf.cell() or pdf.multi_cell() call. A sanitise() helper handles this.

Dependencies
------------
    pip install fpdf2 pillow
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Colour palette (RGB 0-255)
# ---------------------------------------------------------------------------

COL_DARK   = (20,  30,  48)
COL_ACCENT = (56, 140, 180)
COL_LIGHT  = (240, 245, 250)
COL_MUTED  = (120, 130, 140)
COL_WHITE  = (255, 255, 255)
COL_WARN   = (200,  80,  40)


# ---------------------------------------------------------------------------
# Unicode sanitiser
# Helvetica only covers latin-1. Replace common Unicode chars with safe
# ASCII/latin-1 equivalents before passing any string to fpdf2.
# ---------------------------------------------------------------------------

_UNICODE_REPLACEMENTS = {
    # Dashes
    "\u2014": "-",    # em dash       —
    "\u2013": "-",    # en dash       –
    "\u2012": "-",    # figure dash   ‒
    "\u2011": "-",    # non-breaking hyphen
    # Quotes
    "\u2018": "'",    # left single quote  '
    "\u2019": "'",    # right single quote '
    "\u201c": '"',    # left double quote  "
    "\u201d": '"',    # right double quote "
    "\u201e": '"',    # double low-9 quote „
    # Ellipsis
    "\u2026": "...",  # …
    # Bullets / symbols
    "\u2022": "-",    # bullet •
    "\u2023": ">",    # triangular bullet ‣
    "\u25cf": "*",    # black circle ●
    "\u2713": "OK",   # check mark ✓
    "\u2715": "x",    # cross mark ✕
    # Arrows
    "\u2192": "->",   # →
    "\u2190": "<-",   # ←
    "\u21b3": "->",   # ↳
    # Fractions / special
    "\u00b1": "+/-",  # ±
    "\u00d7": "x",    # ×
    "\u00f7": "/",    # ÷
    # Warning / medical symbols that might appear in LLM output
    "\u26a0": "!",    # ⚠
    "\u2764": "<3",   # ❤
}


def sanitise(text: str) -> str:
    """
    Replace Unicode characters outside latin-1 with safe ASCII equivalents,
    then encode/decode through latin-1 to drop anything remaining.
    """
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)

    # Final safety net: encode to latin-1, replacing anything still unsupported
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


# ---------------------------------------------------------------------------
# Helper: numpy array -> temporary JPEG path
# ---------------------------------------------------------------------------

def _array_to_tmp_jpg(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name, format="JPEG", quality=92)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_report_pdf(
    scan_type: str,
    findings: str,
    impression: str,
    recommendation: str,
    model_used: str,
    original_image: Optional[np.ndarray] = None,
    heatmap_image:  Optional[np.ndarray] = None,
    patient_label:  str = "Anonymous",
    extra_metadata: Optional[dict] = None,
) -> bytes:
    """
    Build a PDF report and return raw bytes for st.download_button.

    Parameters
    ----------
    scan_type        : "X-ray" | "CT" | "MRI"
    findings         : LLM findings section text
    impression       : LLM impression section text
    recommendation   : LLM recommendation section text
    model_used       : name of the Ollama model used
    original_image   : RGB uint8 ndarray — scan thumbnail
    heatmap_image    : RGB uint8 ndarray — GradCAM or segmentation overlay
    patient_label    : display label (no real PHI)
    extra_metadata   : additional key-value pairs for the header strip
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf2 is not installed. Run: pip install fpdf2") from exc

    # Sanitise ALL text fields before any PDF call
    scan_type_s      = sanitise(scan_type)
    findings_s       = sanitise(findings)
    impression_s     = sanitise(impression)
    recommendation_s = sanitise(recommendation)
    model_used_s     = sanitise(model_used)
    patient_label_s  = sanitise(patient_label)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(left=15, top=15, right=15)

    # ------------------------------------------------------------------
    # Header bar
    # ------------------------------------------------------------------
    pdf.set_fill_color(*COL_ACCENT)
    pdf.rect(x=0, y=0, w=210, h=28, style="F")

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*COL_WHITE)
    pdf.set_xy(15, 8)
    pdf.cell(0, 10, "Project Consilium - AI Radiology Report", align="L")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_xy(15, 18)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(
        0, 6,
        f"Generated: {timestamp}  |  AI model: {model_used_s}",
        align="L",
    )

    # ------------------------------------------------------------------
    # Metadata strip
    # ------------------------------------------------------------------
    pdf.set_fill_color(*COL_LIGHT)
    pdf.rect(x=0, y=28, w=210, h=16, style="F")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*COL_MUTED)
    pdf.set_xy(15, 32)

    meta_parts = [
        f"Scan type: {scan_type_s}",
        f"Patient: {patient_label_s}",
    ]
    if extra_metadata:
        for k, v in extra_metadata.items():
            meta_parts.append(f"{sanitise(str(k))}: {sanitise(str(v))}")

    pdf.cell(0, 6, "   |   ".join(meta_parts))

    # ------------------------------------------------------------------
    # Disclaimer banner
    # ------------------------------------------------------------------
    pdf.set_fill_color(*COL_WARN)
    pdf.rect(x=0, y=44, w=210, h=10, style="F")
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*COL_WHITE)
    pdf.set_xy(15, 46)
    pdf.cell(
        0, 6,
        "! ASSISTIVE AI ONLY - NOT A SUBSTITUTE FOR PROFESSIONAL MEDICAL DIAGNOSIS",
        align="C",
    )

    y_cursor = 58

    # ------------------------------------------------------------------
    # Scan images (side by side)
    # ------------------------------------------------------------------
    tmp_files: list[str] = []

    if original_image is not None or heatmap_image is not None:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*COL_DARK)
        pdf.set_xy(15, y_cursor)
        pdf.cell(0, 7, "Scan Images")
        y_cursor += 9

        img_w = 85
        img_h = 70

        if original_image is not None:
            path = _array_to_tmp_jpg(original_image)
            tmp_files.append(path)
            pdf.image(path, x=15, y=y_cursor, w=img_w, h=img_h)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*COL_MUTED)
            pdf.set_xy(15, y_cursor + img_h + 1)
            pdf.cell(img_w, 5, "Original scan", align="C")

        if heatmap_image is not None:
            path = _array_to_tmp_jpg(heatmap_image)
            tmp_files.append(path)
            x_pos = 15 + img_w + 10
            pdf.image(path, x=x_pos, y=y_cursor, w=img_w, h=img_h)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*COL_MUTED)
            pdf.set_xy(x_pos, y_cursor + img_h + 1)
            pdf.cell(img_w, 5, "AI attention map / segmentation", align="C")

        y_cursor += img_h + 12

    # ------------------------------------------------------------------
    # Report sections
    # ------------------------------------------------------------------
    sections = [
        ("Findings",       findings_s),
        ("Impression",     impression_s),
        ("Recommendation", recommendation_s),
    ]

    for title, body in sections:
        if not body:
            continue

        # Section title bar
        pdf.set_fill_color(*COL_LIGHT)
        pdf.set_xy(15, y_cursor)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*COL_ACCENT)
        pdf.cell(180, 8, f"  {title}", fill=True, ln=1)
        y_cursor = pdf.get_y() + 2

        # Body text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*COL_DARK)
        pdf.set_xy(15, y_cursor)
        pdf.multi_cell(180, 6, body)
        y_cursor = pdf.get_y() + 6

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    pdf.set_y(-20)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*COL_MUTED)
    report_id = datetime.now().strftime("%Y%m%d%H%M%S")
    pdf.cell(
        0, 5,
        f"Project Consilium research prototype. "
        f"Results must be reviewed by a qualified radiologist before clinical use. "
        f"Report ID: {report_id}",
        align="C",
    )

    # ------------------------------------------------------------------
    # Return bytes and clean up temp image files
    # ------------------------------------------------------------------
    pdf_bytes = pdf.output()

    for path in tmp_files:
        try:
            Path(path).unlink()
        except Exception:
            pass

    return bytes(pdf_bytes)
