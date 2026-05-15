"""
Project Consilium — X-Ray Pipeline
====================================
Uses TorchXRayVision (DenseNet121, CheXpert weights) for
multi-label pathology detection on chest radiographs.

Fixes applied:
  - Input tensor cast to float32
  - GradCAM with use_relu=True
  - Heatmap always generated for top-1 prediction
  - Threshold lowered to 0.20
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from dataclasses import dataclass
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PathologyResult:
    name: str
    score: float

    @property
    def confidence_label(self) -> str:
        if self.score > 0.75:
            return "Strong indication"
        elif self.score > 0.45:
            return "Moderate suspicion"
        elif self.score > 0.20:
            return "Weak evidence"
        return "Below threshold"

    @property
    def confidence_pct(self) -> str:
        return f"{self.score * 100:.1f}%"


@dataclass
class XRayResult:
    all_predictions: list[PathologyResult]
    flagged_predictions: list[PathologyResult]   # above threshold
    heatmap: Optional[np.ndarray]                # RGB uint8 overlay
    top_pathology: str
    top_score: float
    findings_text: str


# ---------------------------------------------------------------------------
# Model loader (cached by caller via @st.cache_resource)
# ---------------------------------------------------------------------------

def load_xray_model() -> xrv.models.DenseNet:
    model = xrv.models.DenseNet(weights="densenet121-res224-chex")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Convert a PIL image to a float32 tensor ready for inference.

    Returns
    -------
    input_tensor : torch.Tensor  shape (1, 1, 224, 224)
    display_array : np.ndarray   shape (224, 224, 3) float32 in [0, 1]
        Normalised grayscale image suitable for show_cam_on_image.
    """
    img_gray = image.convert("L")
    img_array = np.array(img_gray).astype(np.float32)

    # TorchXRayVision normalisation (maps [0,255] → [-1024, 1024])
    img_norm = xrv.datasets.normalize(img_array, 255)
    img_norm = img_norm[None, :, :]    # (1, H, W)

    transform = torchvision.transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])

    img_tensor = transform(img_norm)                      # (1, 224, 224)
    input_tensor = torch.from_numpy(img_tensor).unsqueeze(0).float()  # FIX: .float()

    # Build a normalised [0,1] RGB display image for GradCAM overlay
    display = img_tensor[0]
    display = (display - display.min()) / (display.max() - display.min() + 1e-8)
    display_rgb = np.stack([display, display, display], axis=-1)      # (224, 224, 3)

    return input_tensor, display_rgb


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: xrv.models.DenseNet,
    input_tensor: torch.Tensor,
    threshold: float = 0.20,          # FIX: lowered from 0.75
) -> tuple[list[PathologyResult], list[PathologyResult]]:
    """
    Run forward pass and return (all_predictions, flagged_predictions).
    Results are sorted descending by score.
    """
    with torch.no_grad():
        outputs = model(input_tensor)

    scores = outputs[0].detach().cpu().numpy()
    all_preds = [
        PathologyResult(name=p, score=float(s))
        for p, s in zip(model.pathologies, scores)
    ]
    all_preds.sort(key=lambda x: x.score, reverse=True)

    flagged = [p for p in all_preds if p.score > threshold]
    return all_preds, flagged


# ---------------------------------------------------------------------------
# GradCAM heatmap
# ---------------------------------------------------------------------------

def generate_heatmap(
    model: xrv.models.DenseNet,
    input_tensor: torch.Tensor,
    display_rgb: np.ndarray,
    pathology_name: str,
) -> Optional[np.ndarray]:
    """
    Generate a GradCAM heatmap for the given pathology.
    Always runs on the top-1 prediction so the attention map is
    never suppressed by the detection threshold.

    Returns RGB uint8 ndarray (224, 224, 3) or None on error.

    Notes
    -----
    - use_relu is NOT passed to GradCAM() — it was added in a later
      version of pytorch-grad-cam. We apply ReLU manually on the output
      instead, which is equivalent and works on all versions.
    - cam.activations_and_grads.release() is called explicitly before
      the cam object goes out of scope to avoid the __del__ AttributeError
      that appears in some pytorch-grad-cam versions.
    """
    cam = None
    try:
        disease_idx   = list(model.pathologies).index(pathology_name)
        targets       = [ClassifierOutputTarget(disease_idx)]
        target_layers = [model.features.denseblock4]

        # Do NOT pass use_relu= — not supported in older versions
        cam           = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Apply ReLU manually: clamp negatives to 0 (same effect as use_relu=True)
        grayscale_cam = np.maximum(grayscale_cam, 0)

        # Renormalise to [0, 1] after relu so show_cam_on_image renders correctly
        cam_max = grayscale_cam.max()
        if cam_max > 0:
            grayscale_cam = grayscale_cam / cam_max

        heatmap = show_cam_on_image(display_rgb, grayscale_cam, use_rgb=True)
        return heatmap

    except Exception as exc:
        print(f"[XRay] GradCAM failed for '{pathology_name}': {exc}")
        return None

    finally:
        # Explicitly release hooks to avoid the __del__ AttributeError
        # that some versions of pytorch-grad-cam raise on garbage collection
        if cam is not None:
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Build findings text for LLM
# ---------------------------------------------------------------------------

def build_findings_text(
    flagged: list[PathologyResult],
    top: PathologyResult,
) -> str:
    if flagged:
        parts = [f"{p.name} ({p.confidence_pct})" for p in flagged[:5]]
        return ", ".join(parts)
    return (
        f"No significant abnormalities detected above threshold. "
        f"Highest scored finding: {top.name} ({top.confidence_pct})."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyse_xray(
    model: xrv.models.DenseNet,
    image: Image.Image,
    threshold: float = 0.20,
) -> XRayResult:
    """
    Full X-ray analysis pipeline.

    Parameters
    ----------
    model     : pre-loaded TorchXRayVision DenseNet (use load_xray_model())
    image     : PIL image (any mode; converted internally)
    threshold : confidence threshold for 'flagged' predictions

    Returns
    -------
    XRayResult dataclass with all fields populated
    """
    input_tensor, display_rgb = preprocess_image(image)
    all_preds, flagged = run_inference(model, input_tensor, threshold)

    # FIX: heatmap always generated for top-1, not gated by threshold
    top = all_preds[0]
    heatmap = generate_heatmap(model, input_tensor, display_rgb, top.name)
    findings_text = build_findings_text(flagged, top)

    return XRayResult(
        all_predictions=all_preds,
        flagged_predictions=flagged,
        heatmap=heatmap,
        top_pathology=top.name,
        top_score=top.score,
        findings_text=findings_text,
    )
