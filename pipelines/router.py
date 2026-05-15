"""
Project Consilium — Scan Type Router
======================================
Detects whether an uploaded file is an X-ray, CT scan, or MRI
and routes it to the appropriate analysis pipeline.

Detection strategy (in order of reliability):
  1. DICOM Modality tag (CR, DX → X-ray | CT → CT | MR → MRI)
  2. File extension (.nii, .nii.gz → CT or MRI based on metadata)
  3. User-provided hint (from UI radio button)
  4. Heuristic: single PNG/JPG → treated as X-ray by default
"""

from __future__ import annotations

import io
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Scan type enum
# ---------------------------------------------------------------------------

class ScanType(Enum):
    XRAY = auto()
    CT   = auto()
    MRI  = auto()
    UNKNOWN = auto()

    @property
    def display_name(self) -> str:
        return {
            ScanType.XRAY:    "Chest X-Ray",
            ScanType.CT:      "CT Scan",
            ScanType.MRI:     "MRI Scan",
            ScanType.UNKNOWN: "Unknown",
        }[self]

    @property
    def pipeline_key(self) -> str:
        return {
            ScanType.XRAY:    "xray",
            ScanType.CT:      "ct",
            ScanType.MRI:     "mri",
            ScanType.UNKNOWN: "xray",   # safest fallback
        }[self]


# ---------------------------------------------------------------------------
# DICOM-based detection
# ---------------------------------------------------------------------------

_DICOM_MODALITY_MAP: dict[str, ScanType] = {
    # X-ray modalities
    "CR":  ScanType.XRAY,   # Computed Radiography
    "DX":  ScanType.XRAY,   # Digital X-ray
    "RG":  ScanType.XRAY,   # Radiographic imaging
    "PX":  ScanType.XRAY,   # Panoramic X-Ray
    "XA":  ScanType.XRAY,   # X-Ray Angiography

    # CT modalities
    "CT":  ScanType.CT,

    # MRI modalities
    "MR":  ScanType.MRI,
    "MRA": ScanType.MRI,    # MR Angiography

    # Nuclear medicine (future)
    # "PT": ScanType.PET,
    # "NM": ScanType.SPECT,
}


def detect_from_dicom(raw_bytes: bytes) -> Optional[ScanType]:
    """
    Attempt to read the DICOM Modality tag from raw bytes.
    Returns ScanType or None if not a valid DICOM file.
    """
    try:
        import pydicom
        ds = pydicom.dcmread(io.BytesIO(raw_bytes), stop_before_pixels=True)
        modality = str(getattr(ds, "Modality", "")).upper().strip()
        return _DICOM_MODALITY_MAP.get(modality)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Filename / extension-based detection
# ---------------------------------------------------------------------------

def detect_from_filename(filename: str) -> Optional[ScanType]:
    """
    Use file extension as a secondary detection signal.

    .dcm              → try DICOM header (caller must do this)
    .nii / .nii.gz    → volumetric (CT or MRI — needs header for exact type)
    .png / .jpg / .jpeg → X-ray (single-slice default)
    """
    name  = filename.lower()

    if name.endswith((".nii", ".nii.gz")):
        return None   # ambiguous — need header; caller should inspect further

    if name.endswith(".dcm"):
        return None   # need header

    if name.endswith((".png", ".jpg", ".jpeg")):
        return ScanType.XRAY   # default for plain images

    return None


# ---------------------------------------------------------------------------
# NIfTI header inspection
# ---------------------------------------------------------------------------

def detect_from_nifti(path: str) -> Optional[ScanType]:
    """
    Inspect NIfTI header to distinguish CT from MRI.
    CT headers typically have dim_info relating to slice ordering
    and cal_min/cal_max consistent with HU ranges.
    """
    try:
        import nibabel as nib
        nii = nib.load(path)
        hdr = nii.header

        cal_min = float(hdr.get("cal_min", 0))
        cal_max = float(hdr.get("cal_max", 0))

        # HU range: CT data typically runs from -1024 to 3071
        if cal_min < -500 or cal_max > 1000:
            return ScanType.CT

        # Scl_slope / scl_inter pattern: MRI often has large dynamic range
        return ScanType.MRI

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def detect_scan_type(
    raw_bytes: bytes,
    filename: str = "",
    user_hint: Optional[str] = None,    # "xray" | "ct" | "mri" | None
) -> ScanType:
    """
    Determine the scan type for an uploaded file.

    Priority order:
      1. Explicit user selection (user_hint)
      2. DICOM Modality header
      3. Filename extension
      4. Default: XRAY

    Parameters
    ----------
    raw_bytes  : the uploaded file bytes
    filename   : original filename (used for extension check)
    user_hint  : user's manual selection from UI, if provided

    Returns
    -------
    ScanType enum value
    """
    # 1. Trust the user if they explicitly selected
    if user_hint:
        hint_map = {"xray": ScanType.XRAY, "ct": ScanType.CT, "mri": ScanType.MRI}
        if user_hint.lower() in hint_map:
            return hint_map[user_hint.lower()]

    # 2. Try DICOM header
    dicom_type = detect_from_dicom(raw_bytes)
    if dicom_type is not None:
        return dicom_type

    # 3. Filename extension
    ext_type = detect_from_filename(filename)
    if ext_type is not None:
        return ext_type

    # 4. Default to X-ray for plain image uploads
    return ScanType.XRAY


# ---------------------------------------------------------------------------
# Human-readable detection summary (for UI display)
# ---------------------------------------------------------------------------

def describe_detection(
    scan_type: ScanType,
    method: str,
    filename: str = "",
) -> str:
    """Build a short string explaining how the scan type was determined."""
    method_labels = {
        "user":    "User selected",
        "dicom":   "Detected from DICOM header",
        "filename": "Inferred from filename",
        "default": "Defaulted (no metadata found)",
    }
    label = method_labels.get(method, method)
    name  = scan_type.display_name
    fname = f" ({filename})" if filename else ""
    return f"{label}: **{name}**{fname}"
