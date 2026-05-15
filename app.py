"""
Project Consilium — Main Application (v2)
==========================================
Manual-input multi-modal AI radiology pipeline.

Workflow (exactly as specified)
--------------------------------
1. User uploads scan + manually selects modality (CT/MRI) + target organ.
2. TotalSegmentator crops the target organ from the volume (CT only).
3. Lazy-load the correct custom model into VRAM.
4. Inference → GradCAM heatmap (classifiers) or segmentation overlay (UNets).
5. VRAM cleanup immediately after inference.
6. Pass findings to Ollama LLM → generate clinical report → PDF export.

VRAM safety (RTX 4050, 6 GB)
------------------------------
- Models are NEVER pre-loaded at startup.
- Each model is loaded, used, deleted, and VRAM cleared within one call.
- TotalSegmentator runs with fast=True + roi_subset.
- All inference is inside torch.no_grad().
- CUDA OOM → automatic CPU retry in every analyser.

Run:
    streamlit run app.py

Install:
    pip install streamlit torch torchvision torchxrayvision
    pip install 'monai[nibabel,einops]' pydicom pytorch_grad_cam scipy
    pip install TotalSegmentator fpdf2 pillow numpy ollama opencv-python-headless
    ollama pull phi3:mini
"""

from __future__ import annotations

import gc
import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import os
import multiprocessing

# Kill nnUNet background workers before they are born
os.environ["nnUNet_n_proc_DA"] = "0"
os.environ["nnUNet_compile"] = "f"
multiprocessing.set_start_method("spawn", force=True)

import streamlit as st
from PIL import Image

from pipelines.report_generator import generate_report, get_ollama_status
from pipelines.disease_analyser import (
    run_disease_analysis,
    MODALITY_ORGAN_OPTIONS,
    AnalysisResult,
)
from utils.pdf_exporter import export_report_pdf


# =========================================================
# CONSTANTS / PATHS
# =========================================================

MODELS_DIR = Path(__file__).parent / "models"




# =========================================================
# HELPERS
# =========================================================

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _vram_mb() -> tuple[Optional[float], Optional[float]]:
    try:
        import torch
        if torch.cuda.is_available():
            used  = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return used / 1e6, total / 1e6
    except Exception:
        pass
    return None, None


def _is_nifti(data: bytes) -> bool:
    if len(data) < 4:
        return False
    if data[:2] == b'\x1f\x8b':
        return True
    if len(data) > 348 and data[344:348] in (b'n+1\x00', b'ni1\x00'):
        return True
    return False


# =========================================================
# VOLUME LOADER
# =========================================================

def load_volume(raw_bytes: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """
    Load a scan into float32 (D, H, W).
    Supports NIfTI, DICOM, NPY, PNG/JPG.
    Returns (volume, meta).
    """
    # ── .npy branch ──────────────────────────────────────────────────────────
    if filename.lower().endswith(".npy"):
        try:
            arr = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid MRI format. Please upload a 3D .npy volume. ({exc})"
            ) from exc

        # ── Dimension normalisation ───────────────────────────────────────────
        # Step 1: squeeze leading batch/singleton dims
        #   (1, 155, 240, 240)  → (155, 240, 240)
        #   (1, 1, 64, 64, 64)  → (64, 64, 64)
        arr = np.squeeze(arr)

        # Step 2: handle trailing channel/modality dimension
        #   (128, 128, 128, 3)  → (128, 128, 128)  [keep first channel]
        #   (155, 240, 240, 4)  → (155, 240, 240)  [keep first channel — T1]
        if arr.ndim == 4 and arr.shape[-1] in (1, 2, 3, 4):
            arr = arr[..., 0]   # extract first modality/channel

        # Step 3: validate strictly 3D
        if arr.ndim != 3:
            raise RuntimeError(
                f"Invalid MRI format. Please upload a 3D .npy volume. "
                f"Got shape {arr.shape} after dimension normalisation "
                f"(ndim={arr.ndim}, expected 3). "
                f"Supported layouts: (D,H,W), (1,D,H,W), (D,H,W,C) with C≤4."
            )

        arr = arr.astype(np.float32)
        
        # Save to temp file for TotalSegmentator/Pipeline access
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                np.save(tmp.name, arr)
                tmp_path = tmp.name
        except Exception:
            pass

        # Build a displayable middle-axial-slice preview (uint8 RGB, H×W×3)
        mid   = arr.shape[0] // 2
        sl    = arr[mid].copy()
        lo, hi = float(sl.min()), float(sl.max())
        sl_u8 = ((sl - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        npy_preview = np.stack([sl_u8, sl_u8, sl_u8], axis=-1)   # H×W×3 uint8

        return arr, {
            "spacing":      (1.0, 1.0, 1.0),
            "affine":       np.eye(4),
            "is_volumetric": True,
            "modality_tag": "MRI",
            "tmp_path":     tmp_path,
            "raw_array":    arr,
            "npy_preview":  npy_preview,   # used by UI to show instant preview
        }

    if _is_nifti(raw_bytes):
        import nibabel as nib
        suffix = ".nii.gz" if raw_bytes[:2] == b'\x1f\x8b' else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            nii     = nib.load(tmp_path)
            arr     = nii.get_fdata(dtype=np.float32).transpose(2, 1, 0)
            zooms   = nii.header.get_zooms()
            affine  = nii.affine.copy()
            spacing = tuple(float(z) for z in zooms[:3])
            return arr, {
                "spacing": spacing, "affine": affine,
                "is_volumetric": arr.shape[0] > 1,
                "modality_tag": "CT", "tmp_path": tmp_path,
                "raw_array": arr,
            }
        except Exception as exc:
            Path(tmp_path).unlink(missing_ok=True)
            raise RuntimeError(f"NIfTI load failed: {exc}") from exc

    try:
        import pydicom
        ds    = pydicom.dcmread(io.BytesIO(raw_bytes))
        arr   = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope",    1.0))
        inter = float(getattr(ds, "RescaleIntercept",0.0))
        arr   = arr * slope + inter
        sp    = getattr(ds, "PixelSpacing", [1.0, 1.0])
        spacing = (float(getattr(ds, "SliceThickness", 1.0)),
                   float(sp[0]), float(sp[1]))
        vol = arr[np.newaxis]
        return vol, {
            "spacing": spacing, "affine": np.eye(4),
            "is_volumetric": False,
            "modality_tag": getattr(ds, "Modality", "CT"),
            "tmp_path": None, "raw_array": vol,
        }
    except Exception:
        pass

    img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 255.0) * 1200.0 - 600.0
    vol = arr[np.newaxis]

    # Save to temp file for Universal Loader access
    tmp_path = None
    try:
        suffix = os.path.splitext(filename)[1].lower()
        if not suffix: suffix = ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
    except Exception:
        pass

    return vol, {
        "spacing": (1.0, 1.0, 1.0), "affine": np.eye(4),
        "is_volumetric": False, "modality_tag": "CT",
        "tmp_path": tmp_path, "raw_array": vol,
    }





# =========================================================
# DISPLAY HELPERS
# =========================================================

def get_display_slices(volume: np.ndarray, n: int = 5) -> list[np.ndarray]:
    d       = volume.shape[0]
    indices = np.linspace(0, d - 1, min(n, d), dtype=int)
    slices  = []
    for i in indices:
        sl     = volume[i].astype(np.float32)
        lo, hi = sl.min(), sl.max()
        sl     = ((sl - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        slices.append(np.stack([sl, sl, sl], axis=-1))
    return slices


# =========================================================
# PAGE CONFIG
# =========================================================

def main():
    st.set_page_config(
        page_title="Project Consilium",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
      .findings-box {
        background: rgba(56,160,212,0.08);
        border-left: 4px solid #38a0d4;
        border-radius: 0 8px 8px 0;
        padding: 16px; margin: 8px 0; line-height: 1.7;
      }
      .warning-box {
        background: rgba(224,123,57,0.10);
        border-left: 4px solid #e07b39;
        border-radius: 0 8px 8px 0;
        padding: 12px; margin: 4px 0; font-size:0.9em;
      }
      .result-box {
        background: rgba(40,200,100,0.07);
        border-left: 4px solid #28c864;
        border-radius: 0 8px 8px 0;
        padding: 14px; margin: 8px 0;
      }
      .severe-box {
        background: rgba(220,60,60,0.08);
        border-left: 4px solid #dc3c3c;
        border-radius: 0 8px 8px 0;
        padding: 14px; margin: 8px 0;
      }
    </style>
    """, unsafe_allow_html=True)


    # =========================================================
    # SIDEBAR
    # =========================================================

    with st.sidebar:
        st.markdown("## 🩺 Consilium v2")
        st.markdown("*5-model lazy-loading inference engine*")
        st.markdown("---")

        device    = "cuda" if _cuda_available() else "cpu"
        dev_label = "GPU (CUDA)" if device == "cuda" else "CPU"
        st.caption(f"Device: **{dev_label}**")

        vram_used, vram_total = _vram_mb()
        if vram_total:
            pct = vram_used / vram_total
            st.progress(pct, text=f"VRAM: {vram_used:.0f} / {vram_total:.0f} MB")

        st.markdown("---")
        st.subheader("⚙️ Scan Settings")

        modality = st.radio(
            "Modality",
            options=["CT", "MRI", "X-Ray"],
            index=0,
        )

        if modality == "CT":
            organ = st.selectbox("Target Organ", options=["Brain", "Liver", "Kidney", "Lung"])
        elif modality == "MRI":
            organ = st.selectbox("Target Organ", options=["Brain (Tumor)"])
        else: # X-Ray
            organ = st.selectbox("Target Organ", options=["Lung"])

        st.markdown("---")
        st.subheader("🤖 LLM")
        ollama_status = get_ollama_status()

        if ollama_status["reachable"] and ollama_status["selected"]:
            st.success(f"**{ollama_status['selected']}** ready")
        elif ollama_status["reachable"]:
            st.warning("No preferred model.")
            st.code("ollama pull phi3:mini")
        else:
            st.error("Ollama offline.")
            st.code("ollama serve\nollama pull phi3:mini")

        if ollama_status["models"]:
            with st.expander("Models"):
                for m in ollama_status["models"]:
                    st.text(f"  • {m}")

        st.markdown("**Active model**  \nModel will be auto-selected during analysis based on organ detection.")


    # =========================================================
    # HEADER
    # =========================================================

    st.title("🩺 Project Consilium")
    st.subheader("Manual-Input AI Radiology — VRAM-Safe 5-Model Inference")

    st.info(
        f"**Selected:** `{modality}` — `{organ}`  \n"
        "Upload a scan and click **Run Analysis**."
    )
    st.warning(
        "⚠️ Research prototype — not validated for clinical use. "
        "All findings require radiologist review."
    )


    # =========================================================
    # FILE UPLOADER
    # =========================================================

    uploaded_file = st.file_uploader(
        "Upload scan",
        type=["nii", "gz", "dcm", "png", "jpg", "jpeg", "npy"],
        help="Accepted: NIfTI (.nii/.nii.gz), DICOM (.dcm), Image (.png/.jpg), NumPy (.npy)",
    )

    if uploaded_file is None:
        st.markdown("---")
        cols = st.columns(5)
        for col, (icon, name, arch, desc) in zip(cols, [
            ("🧠", "Brain CT",  "ResNet18",        "Stroke/Normal + GradCAM"),
            ("🔬", "Brain MRI", "AttentionUNet 3D", "Tumour seg + contour"),
            ("🫁", "Lung CT",   "UNet 3D",          "Lesion seg + overlay"),
            ("🫘", "Kidney CT", "DenseNet121",      "4-class + GradCAM"),
            ("🫀", "Liver CT",  "AttentionUNet 3D", "Lesion seg + contour"),
        ]):
            with col:
                st.markdown(f"### {icon} {name}")
                st.caption(f"**{arch}**\n{desc}")
        st.stop()


    # =========================================================
    # LOAD VOLUME
    # =========================================================

    if uploaded_file is not None:
        raw_bytes = uploaded_file.read()
        filename  = uploaded_file.name

        with st.spinner(f"Loading {filename}…"):
            try:
                volume, meta = load_volume(raw_bytes, filename)
            except Exception as exc:
                st.error(f"Could not load scan: {exc}")
                st.stop()

        st.success(
            f"**{filename}** loaded — shape `{volume.shape}` — "
            f"volumetric: {'Yes' if meta['is_volumetric'] else 'No'}"
        )

        # ── NPY preview: show middle-axial slice immediately after upload ──────────
        if meta.get("npy_preview") is not None:
            st.markdown("#### 🔬 MRI Preview (middle axial slice)")
            st.image(
                meta["npy_preview"],
                caption=f"Middle axial slice · shape {volume.shape} · dtype float32",
                width=320,
            )
        else:
            st.markdown("#### Input slices")
            disp_slices = get_display_slices(volume, n=5)
            for col, slc in zip(st.columns(len(disp_slices)), disp_slices):
                with col:
                    st.image(slc, use_container_width=True)


        # =========================================================
        # RUN BUTTON
        # =========================================================

        # =========================================================
        # RUN BUTTON & STATE MANAGEMENT
        # =========================================================

        st.markdown("---")

        if "analyzed_state" not in st.session_state:
            st.session_state.analyzed_state = None
        if "analysis_cache" not in st.session_state:
            st.session_state.analysis_cache = None

        run_clicked = st.button(
            f"🔍 Run Analysis — {modality}",
            type="primary",
            use_container_width=True,
        )

        if run_clicked:
            st.session_state.analyzed_state = (filename, modality)
            st.session_state.analysis_cache = None  # Clear previous cache to force re-run

        if st.session_state.analyzed_state != (filename, modality):
            st.stop()


        # =========================================================
        # ANALYSIS (Manual selection + Lazy load + Infer + VRAM cleanup)
        # =========================================================

        if st.session_state.analysis_cache is None:
            with st.spinner(f"🔍 {modality} {organ} analysis (Lazy load → Infer → VRAM free)…"):
                try:
                    res = run_disease_analysis(
                        volume=volume,
                        modality=modality,
                        organ=organ,
                        device=device,
                        meta=meta,
                    )
                    st.session_state.analysis_cache = res
                    st.success("✅ Inference done — VRAM cleared. Model deleted.")
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    st.session_state.analyzed_state = None
                    st.stop()
                except Exception as exc:
                    st.error(f"Inference error: {exc}")
                    st.session_state.analyzed_state = None
                    st.stop()

        result, seg_map_out, warning, organ_volume = st.session_state.analysis_cache
        crop_applied = organ_volume.shape != volume.shape


        # =========================================================
        # STEP 3 — DISPLAY
        # =========================================================

        st.markdown("---")
        st.subheader(f"📊 Results — {modality} / {result.organ}")

        box_cls = "severe-box" if result.severity in ("severe", "moderate") else "result-box"
        st.markdown(
            f'<div class="{box_cls}">'
            f'{result.icon} <b>{result.organ}</b> [{result.model_key}] — '
            f'<b>{result.label}</b> ({result.confidence:.1%})<br>'
            f'<small>{result.detail}</small>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if result.all_scores:
            st.markdown("**Class probabilities**")
            scols = st.columns(len(result.all_scores))
            for col, (cls, score) in zip(scols, result.all_scores.items()):
                with col:
                    st.metric(cls, f"{score:.1%}")
                    st.progress(float(score))

        # Use the same slice the overlay was drawn on (best_slice_idx = slice with most tumor pixels).
        # Falls back to the middle slice for non-segmentation models (GradCAM etc.).
        _best_idx = result.best_slice_idx
        if _best_idx < 0:
            _best_idx = organ_volume.shape[0] // 2
        _sl = organ_volume[_best_idx].astype(np.float32)
        _lo = np.min(_sl)
        _hi = np.max(_sl)
        if _hi - _lo > 0:
            _sl = (_sl - _lo) / (_hi - _lo)
        else:
            _sl = np.zeros_like(_sl)
        _sl = (_sl * 255).astype(np.uint8)
        mid_slice = np.stack([_sl, _sl, _sl], axis=-1)

        _label = "🔬 Best tumor slice" if result.best_slice_idx >= 0 else "🔬 Original (mid-axial)"
        vis = [(_label, mid_slice)]

        if result.heatmap     is not None: vis.append(("🌡️ GradCAM heatmap",         result.heatmap))
        if result.seg_overlay is not None: vis.append(("🎨 Segmentation overlay",     result.seg_overlay))

        icols = st.columns(len(vis))
        for col, (cap, img) in zip(icols, vis):
            with col:
                st.markdown(f"#### {cap}")
                st.image(img, use_container_width=True)

        for w in result.warnings:
            st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

        vu2, vt2 = _vram_mb()
        if vt2:
            st.caption(f"VRAM after cleanup: {vu2:.0f} / {vt2:.0f} MB")


        # =========================================================
        # FINDINGS + LLM REPORT
        # =========================================================

        findings_text = result.to_llm_text()
        extra_context = (
            f"Modality: {modality}. Organ: {result.organ}. Model: {result.model_key}. "
            f"Shape: {organ_volume.shape}. TotalSeg crop: {crop_applied}."
        )

        st.markdown("---")
        st.subheader("🔍 Findings Summary")
        st.markdown(
            f'<div class="findings-box">{findings_text}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("📝 Clinical Report")

        if not ollama_status["reachable"]:
            st.error("Ollama offline. Run: `ollama serve`  then  `ollama pull phi3:mini`")
        elif not ollama_status["selected"]:
            st.warning("No LLM model. Run: `ollama pull phi3:mini`")
        else:
            gc2, _ = st.columns([1, 3])
            with gc2:
                if st.button(
                    f"Generate report ({ollama_status['selected']})",
                    type="primary",
                    use_container_width=True,
                ):
                    with st.spinner("Generating…"):
                        try:
                            report = generate_report(
                                scan_type=f"{modality} — {result.organ}",
                                findings_text=findings_text,
                                extra_context=extra_context,
                            )
                            st.session_state["consilium_report"] = report
                        except Exception as exc:
                            st.error(f"Report failed: {exc}")

            if "consilium_report" in st.session_state:
                report = st.session_state["consilium_report"]

                for lbl, txt in [
                    ("Findings",       report.findings),
                    ("Impression",     report.impression),
                    ("Recommendation", report.recommendation),
                ]:
                    st.markdown(f"**{lbl}**")
                    st.markdown(f'<div class="findings-box">{txt}</div>', unsafe_allow_html=True)

                st.caption(f"Model: **{report.model_used}**")

                st.markdown("---")
                st.subheader("📥 Export")
                pc, _ = st.columns([1, 3])
                with pc:
                    if st.button("Build PDF", use_container_width=True):
                        with st.spinner("Building PDF…"):
                            try:
                                pdf_bytes = export_report_pdf(
                                    scan_type=f"{modality} — {result.organ}",
                                    findings=report.findings,
                                    impression=report.impression,
                                    recommendation=report.recommendation,
                                    model_used=report.model_used,
                                    original_image=mid_slice,
                                    heatmap_image=(
                                        result.heatmap
                                        if result.heatmap is not None
                                        else result.seg_overlay
                                    ),
                                    extra_metadata={
                                        "File": filename, "Modality": modality,
                                        "Organ": result.organ,   "Model": result.model_key,
                                    },
                                )
                                st.download_button(
                                    "⬇️ Download PDF", pdf_bytes,
                                    file_name=f"consilium_{modality.lower()}_{result.organ.lower()}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True,
                                )
                            except ImportError:
                                st.error("fpdf2 missing: `pip install fpdf2`")
                            except Exception as exc:
                                st.error(f"PDF failed: {exc}")

if __name__ == '__main__':
    main()
