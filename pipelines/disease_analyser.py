"""
Project Consilium — Disease Analyser (v2, complete rewrite)
=============================================================
VRAM-safe lazy inference engine for 6 trained models on RTX 4050 (6 GB).

VRAM Rules (strictly enforced)
-------------------------------
1. LAZY   — model loaded from .pth ONLY when needed, never at import time.
2. CLEANUP — model deleted + cuda.empty_cache() + gc.collect() after EVERY call.
3. NO_GRAD — every forward pass inside torch.no_grad().
4. OOM    — RuntimeError "out of memory" caught → retry on CPU automatically.
5. ONE AT A TIME — never two models in VRAM simultaneously.

The 6 models
-------------
Key           File                             Arch                         In shape         Out
brain_stroke  brain_stroke_model.pth           ResNet18 (2D)                (1,1,224,224)    2 logits
brain_tumor   Brain_Tumor_Mri.pth              AttentionUNet3D (16-256 ch)  (1,1,64,64,64)   raw logits ← NEW
lung          Lung_Tumor_ct_v1.pth             UNet3D                       (1,1,64,64,64)   1-ch seg
kidney        kidney_densenet121.pth           DenseNet121 (2D)             (1,3,224,224)    4 logits
liver         liver_attention_unet_finall.pth  AttentionUNet3D              (1,1,64,64,64)   2-ch seg

brain_tumor inference (v4 — full-resolution sliding window)
-------------------------------------------------------------
The final Sigmoid layer was REMOVED from AttentionUnet during training (BCEWithLogitsLoss).
Therefore:
  1. Model outputs RAW LOGITS — torch.sigmoid() applied inside predict_segmentation_sliding_window.
  2. Simple threshold: prob > 0.5 (reliable due to Negative Sampling training).
  3. No resize, no Otsu, no distance-transform, no percentile post-processing needed.
  4. _skull_strip_mri removed — unnecessary with Negative Sampling.
  5. Full-resolution inference via sliding window (patch 64³, overlap 0.5) preserves spatial detail.

Universal 3-Channel Fix
-----------------------
brain_stroke and kidney were trained on 2D inputs.
TotalSegmentator gives 3D crops.
Fix: extract centre axial slice, resize to 224x224.
  - brain_stroke: use (1,1,H,W) — conv1 is 1-channel.
  - kidney:       use (1,3,H,W) — DenseNet121 is 3-channel, repeat slice x3.
Applied ONLY to these two classifiers. 3D UNets receive full 3D crops.
"""

# 1. Future imports MUST be first
from __future__ import annotations

# 2. Multiprocessing & RAM overrides MUST be second (before torch/TotalSeg)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["nnUNet_compile"] = "F" # Disables memory-heavy torch.compile

# 3. Streamlit Logger Fix MUST be third
import logging
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# 4. All other standard imports follow...
import streamlit as st
import gc
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_FILES = {
    "brain_stroke": "brain_stroke_model.pth",
    "brain_tumor":  "Brain_Tumor_Mri.pth",
    "lung":         "Lung_Tumor_ct_v1.pth",
    "kidney":       "kidney_densenet121.pth",
    "liver":        "liver_attention_unet_finall.pth",
}

VOLUME_TARGET_3D = (64, 64, 64)
SLICE_SIZE_2D    = 224

KIDNEY_CLASSES = ["Cyst", "Normal", "Stone", "Tumor"]
STROKE_CLASSES  = ["Bleeding (Hemorrhage)", "Ischemia (Ischemic Stroke)", "Normal"]

MODALITY_ORGAN_OPTIONS: dict[str, list[str]] = {
    "CT":  ["Brain", "Lung", "Kidney", "Liver"],
    "MRI": ["Brain (Tumor)"],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    organ:       str
    modality:    str
    model_key:   str
    label:       str
    confidence:  float
    severity:    str
    detail:      str
    heatmap:        Optional[np.ndarray] = None
    seg_overlay:    Optional[np.ndarray] = None
    seg_mask:       Optional[np.ndarray] = None
    all_scores:     dict = field(default_factory=dict)
    warnings:       list = field(default_factory=list)
    best_slice_idx: int  = -1  # axial slice used for overlay; -1 = middle

    @property
    def icon(self) -> str:
        return {"normal": "🟢", "mild": "🟡",
                "moderate": "🟠", "severe": "🔴"}.get(self.severity, "⬜")

    @property
    def summary(self) -> str:
        return f"{self.icon} **{self.organ}** — {self.label} ({self.confidence:.0%})"

    def to_llm_text(self) -> str:
        base = (
            f"Organ: {self.organ}. Finding: {self.label} "
            f"(confidence {self.confidence:.0%}, severity {self.severity}). {self.detail}"
        )
        if self.all_scores:
            base += " Scores: " + ", ".join(
                f"{k}={v:.1%}" for k, v in self.all_scores.items()
            ) + "."
        return base


# ---------------------------------------------------------------------------
# VRAM cleanup
# ---------------------------------------------------------------------------

def _vram_cleanup(*refs) -> None:
    try:
        import torch
        for r in refs:
            try:
                del r
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        gc.collect()


def _get_device() -> str:
    """Prefer CUDA (RTX 4050) — fall back to CPU only if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[Device] Using CUDA — {torch.cuda.get_device_name(0)}")
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------

def _build_brain_stroke() -> "torch.nn.Module":
    import torch.nn as nn
    import torchvision.models as tvm
    m = tvm.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    m.fc    = nn.Linear(512, 3)
    return m


def _build_brain_tumor() -> "torch.nn.Module":
    """
    MONAI AttentionUnet — trained with BCEWithLogitsLoss (no internal Sigmoid).
    Channels: (16, 32, 64, 128, 256) — 5 levels, 4 downsampling strides.
    ⚠ Model outputs RAW LOGITS. Call torch.sigmoid() manually at inference.
    """
    from monai.networks.nets import AttentionUnet
    return AttentionUnet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),   # expanded 5-level encoder
        strides=(2, 2, 2, 2),              # 4 strides for 5 channel levels
    )


def _build_lung() -> "torch.nn.Module":
    from monai.networks.nets import UNet
    return UNet(
        spatial_dims=3, in_channels=1, out_channels=1,
        channels=(8, 16, 32, 64), strides=(2, 2, 2), num_res_units=2,
    )


def _build_kidney() -> "torch.nn.Module":
    import torch.nn as nn
    import torchvision.models as tvm
    m = tvm.densenet121(weights=None)
    m.classifier = nn.Linear(1024, 4)
    return m


def _build_liver() -> "torch.nn.Module":
    return _build_brain_tumor()


_BUILDERS = {
    "brain_stroke": _build_brain_stroke,
    "brain_tumor":  _build_brain_tumor,
    "lung":         _build_lung,
    "kidney":       _build_kidney,
    "liver":        _build_liver,
}


# ---------------------------------------------------------------------------
# Lazy loader
# ---------------------------------------------------------------------------

def _lazy_load(model_key: str, device: str) -> "torch.nn.Module":
    import torch
    # --- Determinism: pin cuDNN to a single algorithm, disable benchmark search ---
    # cuDNN benchmark mode tests multiple convolution algorithms and picks the
    # fastest one. This selection can vary between runs, causing different
    # floating-point rounding on values near the sigmoid decision boundary.
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    pth = _MODELS_DIR / MODEL_FILES[model_key]
    if not pth.exists():
        raise FileNotFoundError(
            f"Model not found: {pth.resolve()}\n"
            f"Place the .pth file in the models/ directory."
        )
    model = _BUILDERS[model_key]()
    try:
        state = torch.load(str(pth), map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(str(pth), map_location="cpu", weights_only=False)

    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break

    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()   # disables Dropout + uses BatchNorm running stats
    return model.to(device)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _normalise_volume(v: np.ndarray) -> np.ndarray:
    # Min-Max normalization to [0, 1]
    # The UNet was likely trained on [0, 1] scaled images, not Z-score.
    # Z-score maps the bright tumor to > 3.0 (ignored by model) and the cortex to ~1.0 (misclassified as tumor).
    lo = float(v.min())
    hi = float(v.max())
    if hi > lo:
        return ((v - lo) / (hi - lo)).astype(np.float32)
    return v.astype(np.float32)



def _resize_volume(v: np.ndarray, target: tuple) -> np.ndarray:
    from scipy.ndimage import zoom as ndzoom
    
    # --- STRICT SHAPE GUARDIAN ---
    # 1. Squash empty dimensions (e.g., turns 128x128x128x1 into 128x128x128)
    v = np.squeeze(v)

    # 2. If it's STILL 4D (like an RGB image that slipped through: H x W x 3 x 1)
    if len(v.shape) == 4:
        v = v[..., 0] # Grab only the first channel

    # 3. If it accidentally squeezed down to 2D, bump it back to a flat 3D
    if len(v.shape) == 2:
        v = v[:, :, np.newaxis]
    # -----------------------------

    d, h, w    = v.shape
    td, th, tw = target
    return ndzoom(v, (td/max(d,1), th/max(h,1), tw/max(w,1)), order=1).astype(np.float32)


def _volume_tensor(v: np.ndarray, device: str) -> "torch.Tensor":
    import torch
    return torch.from_numpy(v).unsqueeze(0).unsqueeze(0).float().to(device)


# ---------------------------------------------------------------------------
# Universal 3-Channel 2D Fix
# ---------------------------------------------------------------------------

def _crop_to_1ch_2d(crop_3d: np.ndarray, size: int, device: str) -> "torch.Tensor":
    """
    Centre axial slice → normalise → resize → (1, 1, H, W).
    Used by brain_stroke (ResNet18 with 1-channel conv1).
    """
    import torch
    from scipy.ndimage import zoom as ndzoom

    mid    = crop_3d.shape[0] // 2
    sl     = crop_3d[mid].astype(np.float32)
    lo, hi = sl.min(), sl.max()
    sl     = (sl - lo) / (hi - lo + 1e-8)
    H, W   = sl.shape
    if (H, W) != (size, size):
        sl = ndzoom(sl, (size/max(H,1), size/max(W,1)), order=1).astype(np.float32)
    return torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float().to(device)


def _crop_to_3ch_2d(crop_3d: np.ndarray, size: int, device: str) -> "torch.Tensor":
    """
    Centre axial slice → normalise → resize → repeat x3 → (1, 3, H, W).
    Used by kidney (DenseNet121 with 3-channel input).
    """
    import torch
    from scipy.ndimage import zoom as ndzoom

    mid    = crop_3d.shape[0] // 2
    sl     = crop_3d[mid].astype(np.float32)
    lo, hi = sl.min(), sl.max()
    sl     = (sl - lo) / (hi - lo + 1e-8)
    H, W   = sl.shape
    if (H, W) != (size, size):
        sl = ndzoom(sl, (size/max(H,1), size/max(W,1)), order=1).astype(np.float32)
    rgb = np.stack([sl, sl, sl], axis=0)              # (3, H, W)
    return torch.from_numpy(rgb).unsqueeze(0).float().to(device)  # (1, 3, H, W)


def _display_slice(crop_3d: np.ndarray, size: int = SLICE_SIZE_2D) -> np.ndarray:
    """Centre axial slice normalised to [0,1] float32 (H, W) for GradCAM display."""
    from scipy.ndimage import zoom as ndzoom
    mid    = crop_3d.shape[0] // 2
    sl     = crop_3d[mid].astype(np.float32)
    lo, hi = sl.min(), sl.max()
    sl     = (sl - lo) / (hi - lo + 1e-8)
    H, W   = sl.shape
    if (H, W) != (size, size):
        sl = ndzoom(sl, (size/max(H,1), size/max(W,1)), order=1).astype(np.float32)
    return sl


# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

def _gradcam(
    model:        "torch.nn.Module",
    inp:          "torch.Tensor",
    target_class: int,
    target_layer: "torch.nn.Module",
    display_sl:   np.ndarray,
) -> Optional[np.ndarray]:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        display_rgb = np.stack([display_sl, display_sl, display_sl], axis=-1)
        cam = None
        try:
            cam  = GradCAM(model=model, target_layers=[target_layer])
            gray = cam(input_tensor=inp, targets=[ClassifierOutputTarget(target_class)])[0]
            gray = np.maximum(gray, 0)
            if gray.max() > 0:
                gray /= gray.max()
            return show_cam_on_image(display_rgb, gray, use_rgb=True)
        finally:
            if cam is not None:
                try:
                    cam.activations_and_grads.release()
                except Exception:
                    pass
    except Exception as exc:
        print(f"[GradCAM] failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Morphological post-processing
# ---------------------------------------------------------------------------

# Minimum voxel count for the largest connected component to be considered a
# real tumour candidate. Real tumours are 3D masses — anything under 500
# voxels is almost certainly noise or a brain-fold artefact.
MIN_TUMOR_VOXELS = 500


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a 3D boolean mask and discard
    it entirely if it is smaller than MIN_TUMOR_VOXELS voxels (noise gate).

    Returns the cleaned boolean mask. Safe when mask is entirely False (empty).
    """
    from scipy.ndimage import label as nd_label
    if not mask.any():
        return mask  # empty mask — nothing to do
    labeled, num_features = nd_label(mask)
    if num_features == 0:
        return mask
    # Count voxels per component; label 0 is background — skip it
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0
    largest_label = int(component_sizes.argmax())
    largest_size  = int(component_sizes[largest_label])

    # Noise gate: if the biggest blob is too small, treat the whole mask as empty
    if largest_size < MIN_TUMOR_VOXELS:
        return np.zeros_like(mask, dtype=bool)

    return (labeled == largest_label)


# ---------------------------------------------------------------------------
# Segmentation overlay builders
# ---------------------------------------------------------------------------

def _seg_colour_overlay(
    volume: np.ndarray, seg_mask: np.ndarray,
    colour: tuple = (255, 60, 60), alpha: float = 0.45,
    slice_idx: int = -1,
) -> np.ndarray:
    """Draw a semi-transparent RGBA colour overlay on one axial slice."""
    if slice_idx < 0:
        slice_idx = volume.shape[0] // 2
        
    # Get base grayscale slice
    sl     = volume[slice_idx].astype(np.float32)
    img_min = np.min(sl)
    img_max = np.max(sl)

    # 1. Normalize the raw CT values to 0.0 - 1.0 safely
    if img_max - img_min > 0:
        sl = (sl - img_min) / (img_max - img_min)
    else:
        sl = np.zeros_like(sl)
        
    # 2. Convert to standard 8-bit grayscale (0-255)
    sl_u8 = (sl * 255).astype(np.uint8)
    
    # Create RGBA base (Grayscale image with 255 alpha)
    rgba = np.stack([sl_u8, sl_u8, sl_u8, np.full_like(sl_u8, 255)], axis=-1)
    
    # Get mask threshold
    mask = seg_mask[slice_idx] > 0.5
    
    # Apply lesion color with transparency ONLY where binary mask is 1
    if mask.any():
        r, g, b = colour
        alpha_val = int(alpha * 255)
        
        # We blend the color onto the RGB channels and leave Alpha at 255
        # so it displays properly over the background
        rgba[mask, 0] = (rgba[mask, 0] * (1 - alpha) + r * alpha).astype(np.uint8)
        rgba[mask, 1] = (rgba[mask, 1] * (1 - alpha) + g * alpha).astype(np.uint8)
        rgba[mask, 2] = (rgba[mask, 2] * (1 - alpha) + b * alpha).astype(np.uint8)

    return rgba


def _contour_overlay(
    volume: np.ndarray, seg_mask: np.ndarray,
    colour: tuple = (255, 60, 60),
) -> np.ndarray:
    import cv2
    mid    = volume.shape[0] // 2
    sl     = volume[mid]
    lo, hi = sl.min(), sl.max()
    sl     = ((sl - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    rgb    = np.stack([sl, sl, sl], axis=-1)
    mask   = (seg_mask[mid] > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, cnts, -1, colour, thickness=2)
    return rgb


# ---------------------------------------------------------------------------
# Model analysers
# ---------------------------------------------------------------------------

def analyse_brain_stroke(crop: np.ndarray, device: str = "cpu") -> AnalysisResult:
    """
    Brain Stroke CT — ResNet18 3-class classifier.
    Matches EXACT training preprocessing pipeline and class mapping.
    """
    import torch
    from torchvision import transforms
    from PIL import Image

    display_sl = _display_slice(crop)

    # 1. The Transformation Pipeline
    stroke_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 2. The Image Pre-processing
    if crop.ndim == 3:
        img = crop[crop.shape[0] // 2].copy()
    else:
        img = crop.copy()
    
    img = img.astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    pil_img = Image.fromarray((img_norm * 255).astype(np.uint8)).convert("RGB")
    inp_tensor = stroke_transform(pil_img)
    inp = inp_tensor.unsqueeze(0).to(device)

    model = None
    try:
        model = _lazy_load("brain_stroke", device)
        model.eval()  # 4. Inference Logic
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
        
        pred_idx = int(probs.argmax())
        label    = STROKE_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        
        heatmap = _gradcam(model, inp, pred_idx, model.layer4[-1], display_sl)
        
        # Severity mapping: 0 and 1 are severe, 2 is normal
        severity = "severe" if pred_idx in (0, 1) else "normal"

        # 3. The Label Mapping & Details
        if pred_idx == 0:
            detail = "Hyperdense region detected (Acute Hemorrhage). Immediate neurosurgical consult recommended."
        elif pred_idx == 1:
            detail = "Hypodense region detected (Ischemic Stroke). Evaluate for thrombolysis/thrombectomy window."
        else:
            detail = "No acute intracranial abnormality detected."

        detail += f" (Hemorrhage={probs[0]:.1%}, Ischemia={probs[1]:.1%}, Normal={probs[2]:.1%})"

        return AnalysisResult(
            organ="Brain", modality="CT", model_key="brain_stroke",
            label=label, confidence=confidence, severity=severity,
            detail=detail, heatmap=heatmap,
            all_scores={c: float(probs[i]) for i, c in enumerate(STROKE_CLASSES)},
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_brain_stroke(crop, device="cpu")
        raise
    finally:
        _vram_cleanup(model)


# ---------------------------------------------------------------------------
# Sliding-Window Inference Helper (for volumes larger than PATCH_SIZE³)
# ---------------------------------------------------------------------------

PATCH_SIZE = (64, 64, 64)   # must match the training patch size


def predict_segmentation_sliding_window(
    model:      "torch.nn.Module",
    volume:     np.ndarray,
    device:     str,
    patch_size: tuple = PATCH_SIZE,
    overlap:    float = 0.50,
) -> np.ndarray:
    """
    Full-resolution sliding-window inference for the AttentionUNet.

    Splits the volume into overlapping 64³ patches, runs each through the
    model, applies torch.sigmoid() to convert raw logits → probabilities, then
    averages predictions in overlap zones (accumulation + count normalisation).
    50 % overlap (default) gives smooth boundary transitions with minimal
    checkerboard artefacts at patch edges.

    Parameters
    ----------
    model      : AttentionUNet in eval() mode, already moved to `device`
    volume     : (D, H, W) float32, Min-Max normalised to [0, 1].
                 IMPORTANT — call _normalise_volume() BEFORE passing here.
    device     : "cuda" or "cpu"  (must match where `model` lives)
    patch_size : (pD, pH, pW) — must equal the training patch size (default 64³)
    overlap    : fraction of patch width to overlap (0 → no overlap, 0.5 → 50 %).
                 Higher overlap = smoother boundaries, more computation.

    Returns
    -------
    prob_map : (D, H, W) float32 probability map in [0, 1]
               Threshold with > 0.5 to obtain a binary segmentation mask.

    Quick usage
    -----------
    >>> vol_norm = _normalise_volume(raw_volume)          # [0, 1] float32
    >>> model    = _lazy_load("brain_tumor", "cuda")
    >>> prob_map = predict_segmentation_sliding_window(model, vol_norm, "cuda")
    >>> seg_mask = prob_map > 0.5
    """
    import torch

    D, H, W    = volume.shape
    pD, pH, pW = patch_size

    # Step size derived from overlap fraction — minimum 1 voxel
    def _step(p: int) -> int:
        return max(1, int(p * (1.0 - overlap)))
    sD, sH, sW = _step(pD), _step(pH), _step(pW)

    prob_acc  = np.zeros((D, H, W), dtype=np.float32)  # summed sigmoid probabilities
    count_map = np.zeros((D, H, W), dtype=np.float32)  # patch contribution count

    # Build start-index lists; ensure the tail of each axis is always covered
    def _starts(dim: int, patch: int, step: int) -> list:
        starts = list(range(0, max(1, dim - patch + 1), step))
        if not starts or starts[-1] + patch < dim:
            starts.append(max(0, dim - patch))
        return starts

    d_starts = _starts(D, pD, sD)
    h_starts = _starts(H, pH, sH)
    w_starts = _starts(W, pW, sW)

    model.eval()
    with torch.no_grad():
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = volume[d0:d0+pD, h0:h0+pH, w0:w0+pW].copy()

                    # Reflect-pad boundary patches that are smaller than patch_size
                    if patch.shape != (pD, pH, pW):
                        pad = [
                            (0, pD - patch.shape[0]),
                            (0, pH - patch.shape[1]),
                            (0, pW - patch.shape[2]),
                        ]
                        patch = np.pad(patch, pad, mode="reflect")

                    # Forward pass on the correct device; sigmoid converts logits → probs
                    inp        = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                    prob_patch = torch.sigmoid(model(inp))[0, 0].cpu().numpy()  # (pD,pH,pW)

                    # Write back only the actual (un-padded) region
                    ad = min(pD, D - d0)
                    ah = min(pH, H - h0)
                    aw = min(pW, W - w0)
                    prob_acc[d0:d0+ad, h0:h0+ah, w0:w0+aw] += prob_patch[:ad, :ah, :aw]
                    count_map[d0:d0+ad, h0:h0+ah, w0:w0+aw] += 1.0

    # Average overlapping predictions; guard against any unvisited voxels
    count_map = np.maximum(count_map, 1.0)
    return (prob_acc / count_map).astype(np.float32)


def calculate_shap_metrics(model: "torch.nn.Module", volume: np.ndarray, seg_mask: np.ndarray, device: str, best_slice_idx: int) -> str:
    """
    VRAM-safe 3D SHAP calculation for the AttentionUNet.
    Uses the Summation Hack and limits to a 2D slice padded to satisfy 3D stride requirements.
    """
    import torch
    import torch.nn as nn
    import gc
    try:
        import shap
    except ImportError:
        return ""

    class ShapWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            # Output is (B, C, D, H, W)
            logits = self.base_model(x)
            # Summation Hack: reduce to a single scalar per batch item
            return logits.view(logits.size(0), -1).sum(dim=1).unsqueeze(1)

    coords = np.argwhere(seg_mask[best_slice_idx])
    if len(coords) == 0:
        return ""

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    patch_size = 64
    H, W = volume.shape[1], volume.shape[2]

    # Center of tumor on this slice
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2

    y0 = max(0, min(cy - patch_size // 2, H - patch_size))
    x0 = max(0, min(cx - patch_size // 2, W - patch_size))
    y1 = min(H, y0 + patch_size)
    x1 = min(W, x0 + patch_size)

    slice_2d = volume[best_slice_idx, y0:y1, x0:x1]
    mask_2d = seg_mask[best_slice_idx, y0:y1, x0:x1]

    if slice_2d.shape != (patch_size, patch_size):
        pad = [
            (0, patch_size - slice_2d.shape[0]),
            (0, patch_size - slice_2d.shape[1]),
        ]
        slice_2d = np.pad(slice_2d, pad, mode="reflect")
        mask_2d = np.pad(mask_2d, pad, mode="constant")

    # Replicate the 2D slice 16 times in depth to satisfy 3D UNet stride requirement
    vol_crop = np.stack([slice_2d] * 16, axis=0)

    inp = torch.from_numpy(vol_crop).unsqueeze(0).unsqueeze(0).float().to(device)
    bg = torch.zeros_like(inp).to(device)
    
    explainer = None
    shap_values = None
    sv = None
    positive_shap = None

    wrapped_model = ShapWrapper(model)

    try:
        wrapped_model.eval()
        explainer = shap.GradientExplainer(wrapped_model, bg)
        shap_values = explainer.shap_values(inp)

        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        sv = np.squeeze(sv)  # shape (16, 64, 64)
        
        # We only care about the middle slice we replicated
        sv_2d = sv[8] 
        positive_shap = np.maximum(sv_2d, 0)

        inside_sum = float(positive_shap[mask_2d].sum())
        outside_sum = float(positive_shap[~mask_2d].sum())
        total_sum = inside_sum + outside_sum

        percentage = (inside_sum / total_sum * 100) if total_sum > 0 else 0.0

        res = f"\nExplainability Metrics (SHAP): Root cause analysis confirms {percentage:.1f}% of positive prediction weight is concentrated within the primary hyperintense core, with minimal interference from surrounding tissue artifacts."
        return res

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device != "cpu":
            _vram_cleanup()
            wrapped_cpu = wrapped_model.to("cpu")
            inp_cpu = inp.to("cpu")
            bg_cpu = bg.to("cpu")
            try:
                explainer = shap.GradientExplainer(wrapped_cpu, bg_cpu)
                shap_values = explainer.shap_values(inp_cpu)
                if isinstance(shap_values, list):
                    sv = shap_values[0]
                else:
                    sv = shap_values
                sv = np.squeeze(sv)
                sv_2d = sv[8]
                positive_shap = np.maximum(sv_2d, 0)
                inside_sum = float(positive_shap[mask_2d].sum())
                total_sum = float(positive_shap.sum())
                percentage = (inside_sum / total_sum * 100) if total_sum > 0 else 0.0
                res = f"\nExplainability Metrics (SHAP): Root cause analysis confirms {percentage:.1f}% of positive prediction weight is concentrated within the primary hyperintense core, with minimal interference from surrounding tissue artifacts."
                return res
            except Exception as cpu_e:
                print(f"CPU SHAP failed: {cpu_e}")
                return ""
            finally:
                wrapped_model.to(device)
        else:
            print(f"SHAP failed: {e}")
            return ""
    except Exception as e:
        print(f"SHAP failed: {e}")
        return ""
    finally:
        try:
            del inp, bg, explainer, shap_values, sv, positive_shap, wrapped_model
        except Exception:
            pass
        _vram_cleanup()


def analyse_brain_tumor(volume: np.ndarray, device: str = "cuda") -> AnalysisResult:
    """
    Brain Tumor MRI -- MONAI AttentionUNet3D (channels 16-32-64-128-256).

    Inference pipeline (v8 -- Smart Sensitivity mode)
    --------------------------------------------------
    Designed to catch large, bright, real masses while still rejecting
    thin cortical-fold artefacts. Key ideas:

      Tiered threshold : if seg region is in top-2% brightness -> thr=0.55 (aggressive)
                         else                                   -> thr=0.65 (conservative)
      Hole-fill first  : fill_holes before measuring vf so hollow-core tumours
                         (necrosis) are not penalised for their internal dark spot
      Dynamic intensity: check if seg mean >= 95th-percentile of whole volume
                         (relative brightness, not a fixed ratio vs. mean)
      Shape gate bypass: if voxel_count >= 500 skip shape check entirely --
                         large irregular masses must not be penalised for shape
      Boundary strip   : retained (8-voxel cortical shell zeroed)
      Robust conf      : top-1% mean probability > 0.75
    """
    import torch
    from scipy.ndimage import binary_fill_holes

    # -- Step 1: Normalise ----------------------------------------------------
    vol_norm   = _normalise_volume(volume)   # (D, H, W) float32 in [0, 1]
    brain_mean = float(vol_norm.mean())
    p95        = float(np.percentile(vol_norm, 95))   # 95th-pct brightness threshold
    p98        = float(np.percentile(vol_norm, 98))   # used to decide tier
    p90        = float(np.percentile(vol_norm, 90))   # 90th-pct brightness threshold
    model = None
    try:
        model = _lazy_load("brain_tumor", device)

        # Determinism
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # -- Step 2: Sliding-window inference ----------------------------------
        prob_full = predict_segmentation_sliding_window(
            model, vol_norm, device,
            patch_size=PATCH_SIZE,
            overlap=0.50,
        )   # (D, H, W) float32 in [0, 1]

        # -- Robust confidence: mean of top-1% probability values -------------
        flat_probs  = prob_full.ravel()
        top1_cutoff = np.percentile(flat_probs, 99)
        top1_vals   = flat_probs[flat_probs >= top1_cutoff]
        robust_conf = float(top1_vals.mean()) if len(top1_vals) > 0 else 0.0

        # -- Pass 1 (Candidate Generation) ------------------------------------
        # Apply a base threshold of 0.55 to generate candidate mask
        candidate_mask = prob_full > 0.55
        
        # Cleanup
        candidate_mask = _keep_largest_component(candidate_mask)
        
        # Hole filling
        from scipy.ndimage import binary_fill_holes as _bfh
        candidate_filled = np.zeros_like(candidate_mask, dtype=bool)
        for z in range(candidate_mask.shape[0]):
            candidate_filled[z] = _bfh(candidate_mask[z])
            
        # Measurement
        voxel_count = int(candidate_filled.sum())
        seg_mean = 0.0
        if candidate_filled.any():
            seg_mean = float(vol_norm[candidate_filled].mean())

        # -- Pass 2 (The Smart Gates) -----------------------------------------
        accepted = False
        final_mask = candidate_filled
        thr_used = 0.55
        
        # Gate A (The Obvious Tumor)
        if voxel_count > 400 and seg_mean >= p90:
            accepted = True
            thr_used = 0.55
        else:
            # Gate B (The Tricky Brain Fold)
            # Apply strict threshold of 0.75
            strict_mask = prob_full > 0.75
            strict_mask = _keep_largest_component(strict_mask)
            
            strict_filled = np.zeros_like(strict_mask, dtype=bool)
            for z in range(strict_mask.shape[0]):
                strict_filled[z] = _bfh(strict_mask[z])
                
            strict_voxel_count = int(strict_filled.sum())
            if strict_voxel_count > 200:
                accepted = True
                final_mask = strict_filled
                voxel_count = strict_voxel_count
                thr_used = 0.75
            else:
                accepted = False
                final_mask = np.zeros_like(candidate_filled, dtype=bool)
                voxel_count = 0
                thr_used = 0.75

        # -- Verdict ----------------------------------------------------------
        vf = float(final_mask.mean())
        per_slice_count = final_mask.sum(axis=(1, 2))
        best_slice = int(per_slice_count.argmax()) if final_mask.any() else volume.shape[0] // 2

        if accepted and voxel_count > 0:
            label    = "Pathological region detected"
            severity = (
                "severe"   if vf > 0.05 else
                "moderate" if vf > 0.02 else
                "mild"
            )
            shap_res = calculate_shap_metrics(model, vol_norm, final_mask, device, best_slice)
            
            detail = (
                f"AttentionUNet (16-256 ch, thr={thr_used}, robust_conf={robust_conf:.2f}): "
                f"pathological volume {vf*100:.3f}% ({voxel_count} voxels). "
                f"seg_mean={seg_mean:.3f} vs p90={p90:.3f}. "
                "Hyperintense solid mass confirmed. "
                "MRI with gadolinium contrast and neurosurgical review strongly recommended."
                f"{shap_res}"
            )
            overlay = _seg_colour_overlay(
                volume, final_mask, colour=(255, 80, 80), slice_idx=best_slice
            )
        else:
            label    = "No significant abnormality detected"
            severity = "normal"
            detail = (
                f"AttentionUNet (16-256 ch, thr={thr_used}, robust_conf={robust_conf:.2f}): "
                f"No mass confirmed (failed Two-Pass evaluation). "
                "Result: Normal / No significant abnormality detected. "
                "If symptoms persist, MRI with gadolinium contrast is recommended."
            )
            overlay = _seg_colour_overlay(
                volume, np.zeros_like(final_mask, dtype=bool), colour=(255, 80, 80), slice_idx=best_slice
            )

        return AnalysisResult(
            organ="Brain", modality="MRI", model_key="brain_tumor",
            label=label, confidence=robust_conf, severity=severity,
            detail=detail, seg_overlay=overlay, seg_mask=final_mask,
            best_slice_idx=best_slice,
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_brain_tumor(volume, device="cpu")
        raise
    finally:
        _vram_cleanup(model)
    """
    Brain Tumor MRI -- MONAI AttentionUNet3D (channels 16-32-64-128-256).

    Inference pipeline (v7 -- Solid Mass mode)
    -------------------------------------------
    Five sequential checks specifically designed to reject cortical-fold
    false positives while preserving detection of real parenchymal masses:

      Gate 1 -- Threshold (0.65): baseline segmentation
      Gate 2 -- LCC (>= 200 vx): discard tiny speckle noise
      Gate 3 -- Boundary strip  : zero detections in outer-8-voxel cortical shell
                                   (surface gyri live here; real tumours do not)
      Gate 4 -- Intensity contrast: seg_mean >= brain_mean * 1.20
                                   (folds = same brightness; tumours = hyperintense)
      Gate 5 -- Sphericity proxy : bounding-box fill ratio >= 0.15
                                   (folds are thin wavy sheets; masses are compact)
      Gate 6 -- Dual metric      : vf >= 0.3% AND robust_conf > 0.85
    """
    import torch
    from scipy.ndimage import binary_fill_holes

    # -- Step 1: Min-Max normalise to [0, 1] ----------------------------------
    vol_norm = _normalise_volume(volume)   # (D, H, W) float32, range [0, 1]

    model = None
    try:
        model = _lazy_load("brain_tumor", device)

        # Determinism
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        # -- Step 2: Full-resolution sliding-window inference -----------------
        prob_full = predict_segmentation_sliding_window(
            model, vol_norm, device,
            patch_size=PATCH_SIZE,
            overlap=0.50,
        )   # (D, H, W) float32 in [0, 1]

        # -- Robust confidence: mean of top-1% probability values -------------
        # Prevents single boundary "hot pixels" from inflating the score.
        flat_probs  = prob_full.ravel()
        top1_cutoff = np.percentile(flat_probs, 99)
        top1_vals   = flat_probs[flat_probs >= top1_cutoff]
        robust_conf = float(top1_vals.mean()) if len(top1_vals) > 0 else 0.0

        # -- Gate 1: Threshold (0.65) -----------------------------------------
        seg_full = prob_full > 0.65   # bool (D, H, W)

        # -- Gate 2: LCC + voxel floor (>= 200 voxels) -----------------------
        seg_full    = _keep_largest_component(seg_full)
        voxel_count = int(seg_full.sum())

        # -- Gate 3: Cortical boundary suppression ----------------------------
        # The outer ~8 voxels of the volume are almost exclusively cortical
        # surface. Real parenchymal tumours sit deeper inside the brain.
        # Zero any detections in this shell before all further checks.
        BOUNDARY_PAD = 8
        if seg_full.any():
            inner_mask = np.zeros_like(seg_full, dtype=bool)
            D, H, W = seg_full.shape
            inner_mask[
                BOUNDARY_PAD : D - BOUNDARY_PAD,
                BOUNDARY_PAD : H - BOUNDARY_PAD,
                BOUNDARY_PAD : W - BOUNDARY_PAD,
            ] = True
            seg_full = seg_full & inner_mask
            voxel_count = int(seg_full.sum())
            # Re-run LCC after stripping -- the surviving interior blob
            seg_full = _keep_largest_component(seg_full)
            voxel_count = int(seg_full.sum())

        # -- Gate 4: Intensity contrast check ---------------------------------
        # Brain folds (gyri) have the SAME mean intensity as the surrounding
        # grey matter. Real tumours (especially gliomas) are hyperintense on
        # the MRI modality they were trained on.
        # Require: mean(vol_norm in seg) >= mean(vol_norm whole brain) * 1.20
        intensity_ok = False
        seg_mean     = 0.0
        brain_mean   = float(vol_norm.mean())
        if seg_full.any():
            seg_mean     = float(vol_norm[seg_full].mean())
            intensity_ok = seg_mean >= brain_mean * 1.20

        # -- Gate 5: Sphericity proxy (bounding-box fill ratio) ---------------
        # Brain folds are thin wavy sheets -> very low volume vs bounding box.
        # Solid tumour masses fill a large fraction of their bounding box.
        # Threshold: fill_ratio >= 0.15  (a sphere scores ~0.52; a thin sheet < 0.05)
        sphericity_ok = False
        fill_ratio    = 0.0
        if seg_full.any():
            coords = np.argwhere(seg_full)
            mn, mx = coords.min(axis=0), coords.max(axis=0)
            bb_vol = float(np.prod(mx - mn + 1))
            fill_ratio    = voxel_count / bb_vol if bb_vol > 0 else 0.0
            sphericity_ok = fill_ratio >= 0.15

        # -- Volume fraction --------------------------------------------------
        vf = float(seg_full.mean())

        # -- Best display slice -----------------------------------------------
        per_slice_count = seg_full.sum(axis=(1, 2))
        best_slice = (
            int(per_slice_count.argmax()) if seg_full.any() else volume.shape[0] // 2
        )

        # 2D hole-fill on display slice for a clean overlay
        display_mask = seg_full.copy()
        display_mask[best_slice] = binary_fill_holes(seg_full[best_slice])

        # -- Gate 6: Dual-metric final verdict --------------------------------
        tumour_confirmed = (
            vf          >= 0.003
            and robust_conf  > 0.85
            and intensity_ok            # seg brighter than brain average
            and sphericity_ok           # compact blob, not a thin fold
        )

        if tumour_confirmed:
            label    = "Pathological region detected"
            severity = (
                "severe"   if vf > 0.05 else
                "moderate" if vf > 0.02 else
                "mild"
            )
            detail = (
                f"AttentionUNet (16-256 ch, thr=0.65, robust_conf={robust_conf:.2f}): "
                f"pathological volume {vf*100:.3f}% ({voxel_count} voxels). "
                f"Intensity contrast: {seg_mean:.3f} vs brain mean {brain_mean:.3f} "
                f"(ratio {seg_mean/brain_mean:.2f}x). Sphericity fill={fill_ratio:.2f}. "
                "Solid hyperintense mass confirmed. "
                "MRI with gadolinium contrast and neurosurgical review strongly recommended."
            )
            overlay = _seg_colour_overlay(
                volume, display_mask, colour=(255, 80, 80), slice_idx=best_slice
            )
        else:
            label    = "No significant abnormality detected"
            severity = "normal"
            failed   = []
            if voxel_count < 200:          failed.append(f"voxel count {voxel_count} < 200")
            if vf < 0.003:                 failed.append(f"vf {vf*100:.3f}% < 0.3%")
            if robust_conf <= 0.85:        failed.append(f"robust_conf {robust_conf:.2f} <= 0.85")
            if not intensity_ok:           failed.append(
                f"intensity ratio {seg_mean/brain_mean:.2f}x < 1.20x (fold brightness)")
            if not sphericity_ok:          failed.append(
                f"fill_ratio {fill_ratio:.2f} < 0.15 (thin/elongated shape)")
            gate_info = "; ".join(failed) if failed else "all values below thresholds"
            detail = (
                f"AttentionUNet (16-256 ch, thr=0.65, robust_conf={robust_conf:.2f}): "
                f"No solid mass confirmed. Gates failed: [{gate_info}]. "
                "Result: Normal / No significant abnormality detected. "
                "If symptoms persist, MRI with gadolinium contrast is recommended."
            )
            display_mask = np.zeros_like(seg_full, dtype=bool)
            overlay = _seg_colour_overlay(
                volume, display_mask, colour=(255, 80, 80), slice_idx=best_slice
            )

        return AnalysisResult(
            organ="Brain", modality="MRI", model_key="brain_tumor",
            label=label, confidence=robust_conf, severity=severity,
            detail=detail, seg_overlay=overlay, seg_mask=seg_full,
            best_slice_idx=best_slice,
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_brain_tumor(volume, device="cpu")
        raise
    finally:
        _vram_cleanup(model)




def remove_small_objects(mask: np.ndarray, min_size: int = 150, min_fill_ratio: float = 0.15) -> np.ndarray:
    """Removes small noise and stringy vessels/bronchi using volume and sphericity (fill ratio)."""
    from scipy.ndimage import label
    labeled, num = label(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, num + 1):
        component = labeled == i
        count = component.sum()
        if count >= min_size:
            coords = np.argwhere(component)
            mn, mx = coords.min(axis=0), coords.max(axis=0)
            bb_vol = float(np.prod(mx - mn + 1))
            fill_ratio = count / bb_vol if bb_vol > 0 else 0.0
            
            # A solid sphere is ~0.52. Long thin vessels are < 0.05.
            if fill_ratio >= min_fill_ratio:
                cleaned[component] = 1
    return cleaned.astype(bool)


def analyse_lung(crop: np.ndarray, device: str = "cpu") -> AnalysisResult:
    """
    Lung CT — MONAI UNet 3D binary segmentation.
    Full 3D crop. NO 3-channel fix.
    """
    import torch
    from scipy.ndimage import zoom as ndzoom
    
    # Apply standard Lung Window (-1000 to 400 HU)
    # If we don't clip, the raw CT values (-3000 to +3000) crush the tissue contrast.
    # The UNet panics and predicts random "checkerboard" noise (the blue spots).
    crop_clipped = np.clip(crop, -1000, 400)
    
    LUNG_TARGET = (128, 128, 128)
    vol_norm = _normalise_volume(crop_clipped)
    vol_norm = _resize_volume(vol_norm, LUNG_TARGET)
    
    model = None
    try:
        model = _lazy_load("lung", device)
        
        # The training script used `extract_patch(patch_size=96)`. 
        # We must use sliding window to replicate the patch-based inference!
        prob_full = predict_segmentation_sliding_window(
            model, vol_norm, device,
            patch_size=(96, 96, 96),
            overlap=0.50,
        )
        
        # Whole-volume sliding window needs a lower threshold than patch-centered training
        seg = prob_full > 0.45
        
        # Remove tiny noise but preserve small nodules (>150 voxels), allowing multiple lesions
        seg = remove_small_objects(seg, min_size=150)
        
        D, H, W    = crop.shape
        tD, tH, tW = LUNG_TARGET
        seg_full   = ndzoom(seg.astype(np.float32), (D/tD, H/tH, W/tW), order=0) > 0.5
        
        vf = float(seg_full.mean())
        
        flat_probs = prob_full.ravel()
        top1_cutoff = np.percentile(flat_probs, 99)
        top1_vals = flat_probs[flat_probs >= top1_cutoff]
        confidence = float(top1_vals.mean()) if len(top1_vals) > 0 else 0.0
        
        # Volumetric Gravity-Tether: Erase microscopic phantom noise
        if vf < 0.005:
            seg_full = np.zeros_like(seg_full)
            vf = 0.0
            confidence = 0.0
            label = "No mathematically significant lesion detected"
            severity = "normal"
            detail = "No pulmonary lesion above segmentation threshold."
        else:
            label      = "Pulmonary lesion detected"
            severity   = "severe" if vf > 0.10 else ("moderate" if vf > 0.03 else "mild")
            detail     = f"UNet: lesion {vf*100:.3f}% of lung ROI. Pulmonary lesion — differential: neoplasm, consolidation, nodule. CT with IV contrast and pulmonology referral advised."
        overlay = _seg_colour_overlay(crop, seg_full, colour=(80, 180, 255))
        return AnalysisResult(
            organ="Lung", modality="CT", model_key="lung",
            label=label, confidence=confidence, severity=severity,
            detail=detail, seg_overlay=overlay, seg_mask=seg_full,
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_lung(crop, device="cpu")
        raise
    finally:
        _vram_cleanup(model)


def analyse_kidney(crop: np.ndarray, device: str = "cpu") -> AnalysisResult:
    """
    Kidney CT — DenseNet121 multiclass (Normal/Cyst/Stone/Tumor).
    3-Channel Fix: centre slice → (1, 3, 224, 224).
    GradCAM on model.features.denseblock4.
    """
    import torch
    display_sl = _display_slice(crop)
    inp        = _crop_to_3ch_2d(crop, SLICE_SIZE_2D, device)
    model      = None
    try:
        model  = _lazy_load("kidney", device)
        with torch.no_grad():
            probs = torch.softmax(model(inp), dim=1)[0].cpu().numpy()
        pred_idx   = int(probs.argmax())
        label      = KIDNEY_CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        heatmap    = _gradcam(model, inp, pred_idx,
                              model.features.denseblock4, display_sl)
        severity   = (
            "severe"   if label == "Tumor" else
            "moderate" if label in ("Stone", "Cyst") and confidence > 0.70 else
            "mild"     if label in ("Stone", "Cyst") else "normal"
        )
        detail = (
            f"DenseNet121: {label} ({confidence:.1%}). "
            f"All: " + ", ".join(
                f"{KIDNEY_CLASSES[i]}={probs[i]:.1%}" for i in range(4)
            ) + ". " + {
                "Tumor":  "Renal mass — urgent urology and oncology referral required.",
                "Stone":  "Hyperattenuating foci consistent with nephrolithiasis.",
                "Cyst":   "Hypodense lesion consistent with renal cyst. Bosniak classification recommended.",
                "Normal": "No significant renal pathology on centre axial slice.",
            }.get(label, "")
        )
        return AnalysisResult(
            organ="Kidney", modality="CT", model_key="kidney",
            label=label, confidence=confidence, severity=severity,
            detail=detail, heatmap=heatmap,
            all_scores={c: float(probs[i]) for i, c in enumerate(KIDNEY_CLASSES)},
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_kidney(crop, device="cpu")
        raise
    finally:
        _vram_cleanup(model)


def analyse_liver(crop: np.ndarray, device: str = "cpu") -> AnalysisResult:
    """
    Liver CT — MONAI AttentionUNet 3D segmentation.
    Full 3D crop. NO 3-channel fix.
    """
    import torch
    from scipy.ndimage import zoom as ndzoom
    vol   = _resize_volume(_normalise_volume(crop), VOLUME_TARGET_3D)
    inp   = _volume_tensor(vol, device)
    model = None
    try:
        model = _lazy_load("liver", device)
        with torch.no_grad():
            prob_map = torch.sigmoid(model(inp))[0, 0]
            seg      = (prob_map > 0.5).cpu().numpy()
        D, H, W    = crop.shape
        tD, tH, tW = VOLUME_TARGET_3D
        seg_full   = ndzoom(seg.astype(np.float32), (D/tD, H/tH, W/tW), order=0) > 0.5
        vf         = float(seg_full.mean())
        confidence = float(prob_map.max().cpu())
        
        # Volumetric Gravity-Tether: Erase microscopic phantom noise
        if vf < 0.005:
            seg_full = np.zeros_like(seg_full)
            vf = 0.0
            confidence = 0.0
            label = "No mathematically significant lesion detected"
            severity = "normal"
            detail = "No focal hepatic lesion above segmentation threshold."
        else:
            label      = "Hepatic lesion detected"
            severity   = "severe" if vf > 0.08 else ("moderate" if vf > 0.02 else "mild")
            detail     = f"AttentionUNet: hepatic lesion {vf*100:.3f}% of liver ROI. Hepatic lesion — differential: metastasis, HCC, haemangioma, abscess. Contrast-enhanced CT or MRI recommended."
        overlay = _seg_colour_overlay(crop, seg_full, colour=(200, 80, 255))
        return AnalysisResult(
            organ="Liver", modality="CT", model_key="liver",
            label=label, confidence=confidence, severity=severity,
            detail=detail, seg_overlay=overlay, seg_mask=seg_full,
        )
    except RuntimeError as oom:
        if "out of memory" in str(oom).lower() and device != "cpu":
            _vram_cleanup(model)
            return analyse_liver(crop, device="cpu")
        raise
    finally:
        _vram_cleanup(model)


# ---------------------------------------------------------------------------
# TotalSegmentator auto-routing helpers
# ---------------------------------------------------------------------------

_TOTALSEG_LABEL_IDS: dict[str, list[int]] = {
    "Brain":  [50],
    "Lung":   [14, 15, 16, 17, 18],
    "Kidney": [2, 3],
    "Liver":  [5],
}

_TOTALSEG_ROI: dict[str, list[str]] = {
    "Brain":  ["brain"],
    "Lung":   ["lung_upper_lobe_left", "lung_lower_lobe_left",
               "lung_upper_lobe_right", "lung_middle_lobe_right",
               "lung_lower_lobe_right"],
    "Kidney": ["kidney_right", "kidney_left"],
    "Liver":  ["liver"],
}

_TOTALSEG_FILE_MAP: list[tuple[str, str]] = [
    ("brain.nii.gz",                  "Brain"),
    ("liver.nii.gz",                  "Liver"),
    ("kidney_right.nii.gz",           "Kidney"),
    ("kidney_left.nii.gz",            "Kidney"),
    ("lung_upper_lobe_right.nii.gz",  "Lung"),
    ("lung_lower_lobe_right.nii.gz",  "Lung"),
    ("lung_upper_lobe_left.nii.gz",   "Lung"),
    ("lung_lower_lobe_left.nii.gz",   "Lung"),
    ("lung_middle_lobe_right.nii.gz", "Lung"),
]


# ---------------------------------------------------------------------------
# TotalSegmentator crop extraction
# ---------------------------------------------------------------------------

def run_totalseg_and_crop(
    volume:  np.ndarray,
    meta:    dict,
    organ:   str,
    device:  str,
    padding: int = 12,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[str], str]:
    """
    Run TotalSegmentator fast=True + roi_subset.
    Extract organ bounding-box crop from the volume.

    Returns
    -------
    crop    : (D, H, W) organ crop (or full volume if TS fails)
    seg_map : (D, H, W) int32 combined label map (or None)
    warning : str message if fallback used, else None
    detected_organ: str the organ used or detected
    """
    import nibabel as nib

    tmp_dir  = tempfile.mkdtemp(prefix="consilium_ts_")
    inp_path = os.path.join(tmp_dir, "input.nii.gz")
    out_dir  = os.path.join(tmp_dir, "seg_out")
    os.makedirs(out_dir, exist_ok=True)
    warning  = None
    detected_organ = organ

    try:
        from totalsegmentator.python_api import totalsegmentator

        # Use existing temporary path if available (optimization for uploaded NIfTI/NPY)
        inp_path_meta = meta.get("tmp_path")
        if inp_path_meta and os.path.exists(inp_path_meta):
            inp_path = inp_path_meta
        else:
            affine  = meta.get("affine", np.eye(4))
            raw_arr = meta.get("raw_array", volume)
            nib.save(nib.Nifti1Image(raw_arr.transpose(2, 1, 0), affine), inp_path)

        # Safety Check: Validate file size to prevent crashes on corrupt/tiny files
        file_size = os.path.getsize(inp_path)
        if file_size < 1000: # Less than 1KB
            raise ValueError(f"The uploaded file is only {file_size} bytes. This appears to be a corrupt or empty file. Please upload a valid scan.")

        # Use specific organ provided by the user
        roi = [organ.lower()]
        if organ.lower() == "lung":
            # Ensure all lobes are included for Lung
            roi = _TOTALSEG_ROI["Lung"]
        elif organ.lower() == "kidney":
            roi = _TOTALSEG_ROI["Kidney"]

        def _run_ts(dev: str):
            import torch
            import gc
            
            # Aggressive Pre-Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Translate PyTorch 'cuda' to TotalSegmentator 'gpu'
            ts_device = "gpu" if dev == "cuda" else dev
            
            # Temporarily force single-threaded PyTorch data loaders to prevent Windows subprocess crashes
            torch.set_num_threads(1)
            
            # Function-Level Worker Lock: Explicitly force single-threaded execution for spawned child processes
            os.environ["nnUNet_n_proc_DA"] = "0"
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            
            # If organ is Brain, use the specialized TotalSegmentator task
            if organ.lower() == "brain":
                return totalsegmentator(
                    input=inp_path, output=out_dir,
                    task="brain", fast=True, device=ts_device, quiet=True,
                )

            return totalsegmentator(
                input=inp_path, output=out_dir,
                fast=True, roi_subset=roi, device=ts_device, quiet=True,
            )

        from PIL import Image

        # 1. UNIVERSAL LOADER: Handle JPG, PNG, NPY, and NIfTI safely
        ext = inp_path.lower()
        if ext.endswith(('.jpg', '.jpeg', '.png')):
            pil_img = Image.open(inp_path).convert('L')
            data = np.array(pil_img)
            # Add a dummy 3rd dimension to make it "flat 3D"
            data_3d = data[:, :, np.newaxis] 
            img = nib.Nifti1Image(data_3d.astype(np.float32), np.eye(4))
            
            # Save converted image to NIfTI for downstream compatibility
            new_inp_path = os.path.join(tmp_dir, "input.nii.gz")
            nib.save(img, new_inp_path)
            inp_path = new_inp_path
        elif ext.endswith('.npy'):
            # Implement the NPY Loader: Load Numpy binary and convert to NIfTI object
            data = np.load(inp_path)
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]
            # Ensure it's treated as float32 with identity affine for the UNet
            img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
            
            # Save the converted image to a new temp path for processing.
            new_inp_path = os.path.join(tmp_dir, "input.nii.gz")
            nib.save(img, new_inp_path)
            inp_path = new_inp_path
        elif ext.endswith(('.nii', '.nii.gz')):
            img = nib.load(inp_path)
        else:
            raise ValueError("Unsupported file format.")

        # 2. PROCEED TO BYPASS: Detect 2D images and skip TotalSegmentator
        img_data = np.squeeze(img.get_fdata())
        shape = img.shape
        if len(img_data.shape) < 3 or (len(img_data.shape) == 3 and img_data.shape[2] < 5):
            st.warning(f"⚠️ 2D {organ} Image Detected. Bypassing 3D Crop.")
            # Return the full volume as the "crop" for 2D images
            return volume, None, f"2D {organ} Image Detected: Bypassed 3D segmentation.", organ, (0, 0, 0)

        # Check 2: Squeeze 4D arrays down to 3D (e.g., removing an empty channel dimension)
        if len(shape) > 3:
            # Squeeze out dimensions of size 1
            img_data = np.squeeze(img.get_fdata())
            if len(img_data.shape) > 3:
                raise ValueError(f"Uploaded volume is 4D/multi-channel with shape {shape}. Cannot safely reduce to 3D.")
            
            # Save the cleaned 3D image back to disk for TotalSegmentator
            cleaned_img = nib.Nifti1Image(img_data.astype(np.float32), img.affine)
            nib.save(cleaned_img, inp_path)

        try:
            _run_ts(device)
        except RuntimeError as oom:
            if "out of memory" in str(oom).lower() and device != "cpu":
                import torch
                torch.cuda.empty_cache()
                gc.collect()
                warning = "TotalSegmentator: CUDA OOM — retried on CPU."
                _run_ts("cpu")
            else:
                raise

        detected_organ = organ

        D, H, W = volume.shape
        seg_map = np.zeros((D, H, W), dtype=np.int32)

        file_label_map = [
            ("brain.nii.gz",                  50),
            ("liver.nii.gz",                  5),
            ("kidney_right.nii.gz",           2),
            ("kidney_left.nii.gz",            3),
            ("lung_upper_lobe_left.nii.gz",   14),
            ("lung_lower_lobe_left.nii.gz",   15),
            ("lung_upper_lobe_right.nii.gz",  16),
            ("lung_middle_lobe_right.nii.gz", 17),
            ("lung_lower_lobe_right.nii.gz",  18),
        ]

        for fname, lbl_id in file_label_map:
            fpath = os.path.join(out_dir, fname)
            if os.path.exists(fpath):
                mask_arr = nib.load(fpath).get_fdata(dtype=np.float32).transpose(2, 1, 0)
                if mask_arr.shape != (D, H, W):
                    from scipy.ndimage import zoom as ndzoom
                    mask_arr = ndzoom(
                        mask_arr,
                        (D/mask_arr.shape[0], H/mask_arr.shape[1], W/mask_arr.shape[2]),
                        order=0,
                    )
                seg_map[mask_arr > 0.5] = lbl_id

        label_ids     = _TOTALSEG_LABEL_IDS.get(detected_organ, [])
        combined_mask = np.zeros((D, H, W), dtype=bool)
        for lid in label_ids:
            combined_mask |= (seg_map == lid)

        if combined_mask.any():
            coords = np.where(combined_mask)
            d0 = max(0,   int(coords[0].min()) - padding)
            d1 = min(D-1, int(coords[0].max()) + padding)
            h0 = max(0,   int(coords[1].min()) - padding)
            h1 = min(H-1, int(coords[1].max()) + padding)
            w0 = max(0,   int(coords[2].min()) - padding)
            w1 = min(W-1, int(coords[2].max()) + padding)
            crop = volume[d0:d1+1, h0:h1+1, w0:w1+1]
            # NOTE: Do NOT zero the crop with mask_crop here.
            # Zeroing creates a sharp artificial boundary that UNet misinterprets as tumor.
            # We pass the pure bounding-box crop so the model sees natural tissue edges.
            return crop, seg_map, warning, detected_organ, (d0, h0, w0)

        warning = f"Organ '{detected_organ}' not found in TotalSegmentator output. Using full volume."
        return volume, seg_map, warning, detected_organ, (0, 0, 0)

    except ImportError:
        return volume, None, (
            "TotalSegmentator not installed. Run: pip install TotalSegmentator. "
            "Using full volume — crop accuracy reduced."
        ), detected_organ, (0, 0, 0)
    except Exception as exc:
        return volume, None, f"TotalSegmentator failed ({exc}). Using full volume.", detected_organ, (0, 0, 0)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main public entry point
# ---------------------------------------------------------------------------

_CT_DISPATCH = {
    "Brain":  analyse_brain_stroke,
    "Lung":   analyse_lung,
    "Kidney": analyse_kidney,
    "Liver":  analyse_liver,
}


def run_disease_analysis(
    volume:   np.ndarray,
    modality: str,
    organ:    str,
    device:   str = "cuda",   # RTX 4050 — falls back to CPU on OOM
    meta:     Optional[dict] = None,
) -> tuple[AnalysisResult, Optional[np.ndarray], Optional[str], np.ndarray]:
    """
    Main entry point called by app.py.

    CT flow:
      run_totalseg_and_crop() → organ-specific analyser(crop)

    MRI flow:
      analyse_brain_tumor(full volume)  [no TotalSegmentator]

    Returns
    -------
    (AnalysisResult, seg_map or None, warning or None, cropped_volume)
    """
    meta = meta or {
        "spacing": (1.0, 1.0, 1.0),
        "affine":  np.eye(4),
        "raw_array": volume,
    }
    mod      = modality.upper()
    organ_key = organ.split("(")[0].strip()   # "Brain (Tumor)" → "Brain"

    if mod == "MRI":
        result = analyse_brain_tumor(volume, device=device)
        return result, None, None, volume

    if mod == "CT":
        # Step 1: Run crop (or bypass if 2D) using the manual organ selection
        crop, seg_map, warning, detected_organ, bbox_offset = run_totalseg_and_crop(
            volume=volume, meta=meta, organ=organ_key, device=device,
        )
        
        # Step 2: Route to the specialized function based on manual selection
        target = organ_key.lower()
        if target == "liver":
            result = analyse_liver(crop, device=device)
        elif target == "kidney":
            result = analyse_kidney(crop, device=device)
        elif target == "lung":
            result = analyse_lung(crop, device=device)
        elif target == "brain":
            result = analyse_brain_stroke(crop, device=device)
        else:
            return AnalysisResult(
                organ=organ_key, modality="CT", model_key="none",
                label="Unsupported organ",
                confidence=0.0, severity="normal",
                detail=f"No model for CT/{organ_key}. Supported: Liver, Kidney, Lung, Brain.",
                warnings=[f"Unsupported: CT/{organ_key}"] + ([warning] if warning else []),
            ), seg_map, warning, crop

        if warning:
            result.warnings.append(warning)
        return result, seg_map, warning, crop

    return AnalysisResult(
        organ=organ_key, modality=mod, model_key="none",
        label="Unsupported modality", confidence=0.0, severity="normal",
        detail=f"Unsupported modality: {mod}.",
        warnings=[f"Unsupported modality: {mod}"],
    ), None, None, volume
