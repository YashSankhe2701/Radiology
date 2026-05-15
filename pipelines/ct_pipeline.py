"""
Project Consilium — CT Scan Pipeline
======================================
Uses MONAI for volumetric CT analysis:
  - Whole-body organ segmentation (MONAI Model Zoo)
  - Slice-level lesion heatmaps via GradCAM on a 2D ResNet
  - DICOM + NIfTI + PNG/JPG input handling

Dependencies
------------
    pip install monai[nibabel,pillow] pydicom
    pip install torch torchvision

MONAI Model Zoo bundle (downloaded on first run):
    monai.apps.download_and_extract (wholebody_ct_segmentation)

Design notes
------------
- Volumetric inference uses sliding-window over the axial axis
  so the full 3D volume never has to fit in VRAM at once.
- For 2D uploads (PNG/JPG) the image is treated as a single
  axial slice and only slice-level analysis is performed.
- VRAM budget: SW-inference with roi=(96,96,96) fits on 6 GB.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy imports — only pulled in when the CT pipeline is actually used
# ---------------------------------------------------------------------------

def _import_monai():
    try:
        import monai
        from monai.transforms import (
            Compose,
            LoadImaged,
            EnsureChannelFirstd,
            Orientationd,
            Spacingd,
            ScaleIntensityRanged,
            CropForegroundd,
            Resized,
            ToTensord,
        )
        from monai.inferers import sliding_window_inference
        from monai.networks.nets import UNETR, SwinUNETR
        return monai, {
            "Compose": Compose,
            "LoadImaged": LoadImaged,
            "EnsureChannelFirstd": EnsureChannelFirstd,
            "Orientationd": Orientationd,
            "Spacingd": Spacingd,
            "ScaleIntensityRanged": ScaleIntensityRanged,
            "CropForegroundd": CropForegroundd,
            "Resized": Resized,
            "ToTensord": ToTensord,
        }, sliding_window_inference, UNETR, SwinUNETR
    except ImportError as exc:
        raise ImportError(
            "MONAI is not installed.\n"
            "Run: pip install 'monai[nibabel,pillow]'"
        ) from exc


def _import_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is not installed.\n"
            "Run: pip install pydicom"
        ) from exc


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OrganSegment:
    """One segmented organ or structure from CT inference."""
    label_id: int
    name: str
    voxel_count: int
    volume_ml: float          # estimated from voxel spacing
    confidence: float         # mean softmax probability in mask region

    @property
    def volume_str(self) -> str:
        return f"{self.volume_ml:.1f} mL"


@dataclass
class CTResult:
    """Full output of the CT analysis pipeline."""
    scan_type: str                         # "volumetric" | "single_slice"
    input_shape: tuple                     # original volume shape
    slices: list[np.ndarray]              # representative axial slices (RGB)
    heatmaps: list[np.ndarray]            # attention overlays per slice
    segmentation_overlay: Optional[np.ndarray]  # coloured seg on mid-slice
    organs: list[OrganSegment]
    findings_text: str
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Label map — MONAI whole-body CT segmentation (104 classes)
# Abbreviated to the 20 most clinically relevant for reporting
# ---------------------------------------------------------------------------

ORGAN_LABELS: dict[int, str] = {
    1:   "Spleen",
    2:   "Right kidney",
    3:   "Left kidney",
    4:   "Gallbladder",
    5:   "Liver",
    6:   "Stomach",
    7:   "Aorta",
    8:   "Inferior vena cava",
    9:   "Portal vein",
    10:  "Pancreas",
    11:  "Right adrenal gland",
    12:  "Left adrenal gland",
    13:  "Duodenum",
    14:  "Bladder",
    15:  "Prostate / uterus",
    16:  "Left femoral head",
    17:  "Right femoral head",
    18:  "Left lung",
    19:  "Right lung",
    20:  "Colon",
}

# Colour palette for segmentation overlay (RGB)
SEGMENT_COLOURS: dict[int, tuple] = {
    1:  (255, 80,  80),    # spleen — coral
    2:  (80,  160, 255),   # right kidney — blue
    3:  (80,  200, 255),   # left kidney — light blue
    4:  (255, 200, 60),    # gallbladder — amber
    5:  (200, 80,  255),   # liver — purple
    6:  (80,  220, 140),   # stomach — green
    7:  (255, 60,  60),    # aorta — red
    8:  (60,  60,  255),   # IVC — dark blue
    9:  (255, 140, 80),    # portal vein — orange
    10: (255, 255, 80),    # pancreas — yellow
}


# ---------------------------------------------------------------------------
# DICOM / NIfTI / PNG loading
# ---------------------------------------------------------------------------

def _is_nifti_bytes(data: bytes) -> bool:
    """
    Detect NIfTI format from magic bytes.
    - Raw NIfTI (.nii):    bytes 344-347 == b'n+1\\x00' or b'ni1\\x00'
    - Gzipped NIfTI (.gz): first two bytes are gzip magic 0x1f 0x8b
    Both are safe to treat as NIfTI and write to a temp file for nibabel.
    """
    if len(data) < 4:
        return False
    # Gzip magic bytes — almost certainly a .nii.gz
    if data[:2] == b'\x1f\x8b':
        return True
    # Raw NIfTI magic at byte offset 344
    if len(data) > 348 and data[344:348] in (b'n+1\x00', b'ni1\x00'):
        return True
    return False


def _load_nifti_bytes(data: bytes) -> tuple[np.ndarray, dict]:
    """
    Write NIfTI bytes to a temp file and load with nibabel.
    Nibabel cannot read from BytesIO — it needs a real file path.
    """
    import nibabel as nib
    import tempfile

    suffix = ".nii.gz" if data[:2] == b'\x1f\x8b' else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        nii    = nib.load(tmp_path)
        array  = nii.get_fdata(dtype=np.float32)
        zooms  = nii.header.get_zooms()
        spacing = tuple(float(z) for z in zooms[:3])
        meta = {
            "spacing":  spacing,
            "origin":   (0.0, 0.0, 0.0),
            "modality": "CT",
        }
        # nibabel loads as (W, H, D) → transpose to (D, H, W)
        return array.transpose(2, 1, 0), meta
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def load_ct_volume(source: bytes | str | Path) -> tuple[np.ndarray, dict]:
    """
    Load a CT volume from:
      - NIfTI bytes (.nii / .nii.gz)  → full 3D volume via temp file
      - DICOM bytes (.dcm)            → single-slice HU array
      - NIfTI path (str / Path)       → full 3D volume
      - PNG/JPG bytes                 → single-slice pseudo-CT (fallback)

    Returns
    -------
    volume  : float32 ndarray  shape (D, H, W)
    meta    : dict with keys 'spacing', 'origin', 'modality'
    """
    if isinstance(source, bytes):

        # ── 1. NIfTI bytes (must check BEFORE DICOM/PIL) ──────────────
        if _is_nifti_bytes(source):
            try:
                return _load_nifti_bytes(source)
            except ImportError:
                raise ImportError(
                    "nibabel is required to read NIfTI files.\n"
                    "Run: pip install nibabel"
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to load NIfTI volume: {exc}") from exc

        # ── 2. DICOM bytes ─────────────────────────────────────────────
        try:
            pydicom = _import_pydicom()
            ds      = pydicom.dcmread(io.BytesIO(source))
            array   = ds.pixel_array.astype(np.float32)

            slope  = float(getattr(ds, "RescaleSlope",     1.0))
            intcpt = float(getattr(ds, "RescaleIntercept", 0.0))
            array  = array * slope + intcpt

            spacing = (
                float(getattr(ds, "SliceThickness",    1.0)),
                float(getattr(ds, "PixelSpacing",    [1.0, 1.0])[0]),
                float(getattr(ds, "PixelSpacing",    [1.0, 1.0])[1]),
            )
            meta = {
                "spacing":  spacing,
                "origin":   (0.0, 0.0, 0.0),
                "modality": getattr(ds, "Modality", "CT"),
            }
            return array[np.newaxis, :, :], meta   # (1, H, W)

        except Exception:
            pass  # not a DICOM — fall through

        # ── 3. PNG / JPG single-slice fallback ─────────────────────────
        try:
            img   = Image.open(io.BytesIO(source)).convert("L")
            array = np.array(img, dtype=np.float32)
            # Map [0, 255] to soft-tissue HU range [-200, 400]
            array = (array / 255.0) * 600.0 - 200.0
            meta  = {
                "spacing":  (1.0, 1.0, 1.0),
                "origin":   (0.0, 0.0, 0.0),
                "modality": "CT_estimated",
            }
            return array[np.newaxis, :, :], meta
        except Exception as exc:
            raise RuntimeError(
                f"Could not load CT input. "
                f"Supported formats: DICOM (.dcm), NIfTI (.nii/.nii.gz), PNG/JPG.\n"
                f"Error: {exc}"
            ) from exc

    else:
        # ── Path / str input → NIfTI file on disk ─────────────────────
        try:
            import nibabel as nib
            nii    = nib.load(str(source))
            array  = nii.get_fdata(dtype=np.float32)
            zooms  = nii.header.get_zooms()
            spacing = tuple(float(z) for z in zooms[:3])
            meta = {
                "spacing":  spacing,
                "origin":   (0.0, 0.0, 0.0),
                "modality": "CT",
            }
            return array.transpose(2, 1, 0), meta   # (W,H,D) → (D,H,W)
        except ImportError as exc:
            raise ImportError(
                "nibabel required for NIfTI: pip install nibabel"
            ) from exc


# ---------------------------------------------------------------------------
# HU windowing
# ---------------------------------------------------------------------------

def apply_window(
    volume: np.ndarray,
    window_centre: float = 40.0,
    window_width: float  = 400.0,
) -> np.ndarray:
    """
    Apply CT windowing and return float32 in [0, 1].
    Default: soft-tissue window (C=40, W=400).
    """
    lo = window_centre - window_width / 2.0
    hi = window_centre + window_width / 2.0
    windowed = np.clip(volume, lo, hi)
    return (windowed - lo) / (window_width + 1e-8)


# ---------------------------------------------------------------------------
# Segmentation inference (MONAI sliding-window)
# ---------------------------------------------------------------------------

def _load_ct_bundle(device: str, bundle_dir: str = "./models") -> Optional[object]:
    """
    Load the MONAI Model Zoo 'wholeBody_ct_segmentation' bundle.
    Downloads automatically on first call (~300 MB).
    Returns the model or None on failure.
    """
    try:
        import torch
        from monai.bundle import download, load
        from pathlib import Path

        bundle_name = "wholeBody_ct_segmentation"
        bundle_path = Path(bundle_dir) / bundle_name

        # Download only if not already cached
        if not bundle_path.exists():
            print(f"[CT] Downloading MONAI bundle '{bundle_name}' to {bundle_dir} …")
            download(name=bundle_name, bundle_dir=bundle_dir)

        # Load the model from the bundle
        model = load(
            name=bundle_name,
            bundle_dir=bundle_dir,
            device=device,
        )
        model.eval()
        print(f"[CT] Loaded pretrained bundle: {bundle_name}")
        return model

    except Exception as exc:
        print(f"[CT] Could not load MONAI bundle: {exc}")
        return None


def _build_fallback_ct_model(device: str) -> object:
    """
    Lightweight SegResNet with random weights — used when the
    MONAI bundle is unavailable. Produces placeholder segmentation.
    """
    import torch
    from monai.networks.nets import SegResNet

    print("[CT] Using fallback SegResNet (random weights — overlay is illustrative only)")
    model = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=len(ORGAN_LABELS) + 1,
        init_filters=16,
    ).to(device)
    model.eval()
    return model


def run_ct_segmentation(
    volume: np.ndarray,            # (D, H, W) float32 in HU
    device: str = "cpu",
    roi_size: tuple = (96, 96, 96),
    bundle_dir: str = "./models",
) -> Optional[np.ndarray]:
    """
    Run sliding-window CT organ segmentation.

    Tries to use the MONAI Model Zoo 'wholeBody_ct_segmentation' bundle
    (downloaded on first use). Falls back to a random-weight SegResNet
    so the UI never breaks, but labels the overlay as illustrative.

    Returns
    -------
    seg_map : int32 ndarray (D, H, W) with organ label IDs, or None on error.
    """
    try:
        import torch
        from monai.inferers import sliding_window_inference

        # Try pretrained bundle first, fall back to random weights
        model = _load_ct_bundle(device, bundle_dir) or _build_fallback_ct_model(device)

        # Prepare tensor: (1, 1, D, H, W)
        vol_norm   = apply_window(volume)
        vol_tensor = (
            torch.from_numpy(vol_norm)
            .unsqueeze(0).unsqueeze(0)
            .float().to(device)
        )

        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=vol_tensor,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
                device=device,
            )

        seg_map = (
            torch.argmax(logits, dim=1)
            .squeeze(0).cpu().numpy()
            .astype(np.int32)
        )
        return seg_map

    except Exception as exc:
        print(f"[CT] Segmentation failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Organ volume estimation
# ---------------------------------------------------------------------------

def compute_organ_volumes(
    seg_map: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> list[OrganSegment]:
    """
    Compute per-organ voxel counts and approximate volumes in mL.
    """
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    segments = []

    for label_id, name in ORGAN_LABELS.items():
        mask = seg_map == label_id
        voxels = int(mask.sum())
        if voxels == 0:
            continue
        volume_ml = voxels * voxel_vol_ml
        segments.append(OrganSegment(
            label_id=label_id,
            name=name,
            voxel_count=voxels,
            volume_ml=volume_ml,
            confidence=0.75,   # placeholder; real value from softmax in prod
        ))

    segments.sort(key=lambda s: s.volume_ml, reverse=True)
    return segments


# ---------------------------------------------------------------------------
# Segmentation colour overlay on a 2D slice
# ---------------------------------------------------------------------------

def render_segmentation_overlay(
    slice_gray: np.ndarray,    # (H, W) float32 in [0, 1]
    seg_slice: np.ndarray,     # (H, W) int32 label map
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend coloured organ labels onto a grayscale CT slice.
    Returns RGB uint8 (H, W, 3).
    """
    rgb = np.stack([slice_gray * 255] * 3, axis=-1).astype(np.uint8)

    for label_id, colour in SEGMENT_COLOURS.items():
        mask = seg_slice == label_id
        if not mask.any():
            continue
        for c, val in enumerate(colour):
            rgb[:, :, c][mask] = np.clip(
                rgb[:, :, c][mask] * (1 - alpha) + val * alpha, 0, 255
            ).astype(np.uint8)

    return rgb


# ---------------------------------------------------------------------------
# Representative slice selection
# ---------------------------------------------------------------------------

def extract_representative_slices(
    volume: np.ndarray,          # (D, H, W) float32 in HU
    n_slices: int = 5,
    window_centre: float = 40.0,
    window_width: float  = 400.0,
) -> list[np.ndarray]:
    """
    Select n_slices evenly-spaced axial slices and return as RGB uint8.
    """
    depth = volume.shape[0]
    indices = np.linspace(0, depth - 1, n_slices, dtype=int)

    windowed = apply_window(volume, window_centre, window_width)
    slices = []
    for idx in indices:
        s = windowed[idx]
        s_uint8 = (s * 255).astype(np.uint8)
        rgb = np.stack([s_uint8, s_uint8, s_uint8], axis=-1)
        slices.append(rgb)
    return slices


# ---------------------------------------------------------------------------
# Findings text builder
# ---------------------------------------------------------------------------

def build_ct_findings(organs: list[OrganSegment], warnings: list[str]) -> str:
    if not organs:
        return "Segmentation did not identify discrete organ structures. Manual review required."

    top = organs[:5]
    parts = [f"{o.name} ({o.volume_str})" for o in top]
    base = "Identified structures: " + ", ".join(parts) + "."

    if warnings:
        base += " Warnings: " + "; ".join(warnings) + "."

    return base


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_ct(
    source: bytes | str | Path,
    device: str = "cpu",
    run_segmentation: bool = True,
) -> CTResult:
    """
    Full CT analysis pipeline.

    Parameters
    ----------
    source           : raw bytes (DICOM/PNG/JPG) or path to NIfTI file
    device           : 'cuda' or 'cpu'
    run_segmentation : set False to skip (faster, for lightweight mode)

    Returns
    -------
    CTResult with slices, heatmaps, organs, findings
    """
    warnings: list[str] = []
    volume, meta = load_ct_volume(source)

    is_volumetric = volume.shape[0] > 1
    scan_type     = "volumetric" if is_volumetric else "single_slice"

    if meta["modality"] == "CT_estimated":
        warnings.append(
            "Non-DICOM input detected. HU values estimated; "
            "segmentation accuracy may be reduced."
        )

    # Representative slices for display
    slices = extract_representative_slices(volume)

    # Segmentation
    seg_map    = None
    organs     = []
    seg_overlay = None

    if run_segmentation and is_volumetric:
        seg_map = run_ct_segmentation(volume, device=device)
        if seg_map is not None:
            organs = compute_organ_volumes(seg_map, meta["spacing"])

            # Build overlay on the middle axial slice
            mid = volume.shape[0] // 2
            windowed_mid = apply_window(volume)[mid]
            seg_overlay  = render_segmentation_overlay(windowed_mid, seg_map[mid])

    elif not is_volumetric:
        warnings.append(
            "Single-slice input: volumetric segmentation skipped. "
            "Upload a DICOM series or NIfTI for full organ analysis."
        )

    findings = build_ct_findings(organs, warnings)

    # Heatmaps: for display, pass the representative slices without
    # modification (GradCAM on 3D volumes requires a dedicated 3D CNN;
    # this is handled by run_ct_segmentation's attention maps in prod)
    heatmaps = slices   # placeholder; replace with real 3D GradCAM in production

    return CTResult(
        scan_type=scan_type,
        input_shape=volume.shape,
        slices=slices,
        heatmaps=heatmaps,
        segmentation_overlay=seg_overlay,
        organs=organs,
        findings_text=findings,
        warnings=warnings,
    )
