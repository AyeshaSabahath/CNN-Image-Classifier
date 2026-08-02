"""
Streamlit web application for CNN Image Classification.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from predict import (
    capture_webcam_frame,
    load_metadata,
    load_model,
    predict_array,
    predict_with_gradcam,
)
from utils import (
    DATASET_CHOICE,
    IMAGES_DIR,
    MODEL_PATH,
    MODELS_DIR,
    get_dataset_config,
    setup_logging,
)

logger = setup_logging("app")

st.set_page_config(
    page_title="Image Classification using CNN",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0f766e 0%, #134e4a 45%, #1e293b 100%);
    padding: 2rem 2.25rem;
    border-radius: 18px;
    color: #f8fafc;
    margin-bottom: 1.5rem;
    box-shadow: 0 12px 40px rgba(15, 118, 110, 0.25);
}
.main-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.main-header p {
    margin: 0;
    opacity: 0.9;
    font-size: 1.05rem;
}

.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}
.metric-chip {
    display: inline-block;
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
    border-radius: 10px;
    padding: 0.45rem 0.85rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}
.result-banner {
    background: linear-gradient(120deg, #ecfdf5, #f0fdfa);
    border-left: 5px solid #0f766e;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0 1rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #0f766e, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
}
.stButton > button:hover {
    filter: brightness(1.05);
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.35);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Load the Keras model once per session/process."""
    return load_model(MODEL_PATH)


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "webcam_frame" not in st.session_state:
        st.session_state.webcam_frame = None


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>Image Classification using CNN</h1>
            <p>Upload an image, capture from webcam, and inspect Grad-CAM explanations —
            powered by TensorFlow / Keras.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    """Sidebar with project info and dataset context."""
    meta = {}
    meta_path = MODELS_DIR / "training_metadata.json"
    metrics_path = MODELS_DIR / "evaluation_metrics.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = get_dataset_config(DATASET_CHOICE)
        meta["dataset"] = meta.get("name", DATASET_CHOICE)

    eval_metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            eval_metrics = json.load(f)

    with st.sidebar:
        st.markdown("### Project Information")
        st.write(
            "A modular Convolutional Neural Network pipeline for real-world "
            "image classification with training, evaluation, and interactive inference."
        )

        st.markdown("### Dataset Selection")
        dataset_name = meta.get("dataset") or meta.get("name") or DATASET_CHOICE
        st.info(f"Active training dataset: **{dataset_name}**")
        st.caption(
            "Change `DATASET_CHOICE` in `utils.py` (`fashion_mnist` or `cats_dogs`), "
            "then re-run `python train.py`."
        )

        st.markdown("### About CNN")
        st.write(
            "Convolutional Neural Networks learn hierarchical visual features "
            "via Conv → ReLU → Pooling stacks, then classify with dense layers."
        )
        st.code(
            "Conv32 → Pool → Conv64 → Pool → Conv128 → Pool\n"
            "→ Flatten → Dense(256) → Dropout → Output",
            language="text",
        )

        st.markdown("### Model Accuracy")
        if eval_metrics:
            st.metric("Test Accuracy", f"{eval_metrics.get('accuracy', 0) * 100:.2f}%")
            st.metric("F1 Score", f"{eval_metrics.get('f1_score', 0) * 100:.2f}%")
        elif meta.get("best_val_accuracy") is not None:
            st.metric("Best Val Accuracy", f"{float(meta['best_val_accuracy']) * 100:.2f}%")
        else:
            st.warning("No trained model metrics yet. Run `python train.py`.")

        model_exists = MODEL_PATH.exists()
        st.markdown(
            f"<span class='metric-chip'>{'Model Ready' if model_exists else 'Model Missing'}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.caption("Tech: Python · TensorFlow · Streamlit · OpenCV · scikit-learn")

    return meta


def plot_probability_chart(probabilities: dict):
    """Horizontal bar chart of class probabilities."""
    names = list(probabilities.keys())
    values = [probabilities[n] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.35 * len(names) + 1)))
    cmap = plt.get_cmap("Greens")
    colors = cmap(np.linspace(0.35, 0.85, len(names)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Probability (%)")
    ax.set_xlim(0, 100)
    ax.set_title("Class Probabilities")
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(min(v + 1.5, 95), i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def append_history(result: dict, source: str) -> None:
    st.session_state.history.insert(
        0,
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence_pct"],
            "inference_time_ms": result["inference_time_ms"],
        },
    )
    # Keep a reasonable history length
    st.session_state.history = st.session_state.history[:50]
    st.session_state.last_result = result


def build_report_text(result: dict, source: str) -> str:
    lines = [
        "Image Classification CNN — Prediction Report",
        "=" * 48,
        f"Timestamp      : {datetime.now().isoformat(timespec='seconds')}",
        f"Source         : {source}",
        f"Dataset        : {result.get('dataset', 'n/a')}",
        f"Predicted Class: {result['predicted_class']}",
        f"Confidence     : {result['confidence_pct']}",
        f"Inference Time : {result['inference_time_ms']} ms",
        "",
        "Probabilities:",
    ]
    for name, prob in result.get("probabilities", {}).items():
        lines.append(f"  - {name}: {prob * 100:.2f}%")
    lines.append("")
    lines.append("Generated by Image Classification using CNN")
    return "\n".join(lines)


def show_model_summary(model) -> None:
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    st.code(stream.getvalue(), language="text")


def run_prediction_ui(image_np: np.ndarray, source: str, show_gradcam: bool) -> None:
    """Shared prediction display for upload / webcam paths."""
    try:
        model = get_cached_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    with st.spinner("Running inference..."):
        if show_gradcam:
            result, overlay = predict_with_gradcam(image_np, model=model)
        else:
            result = predict_array(image_np, model=model)
            overlay = None

    append_history(result, source)

    st.markdown(
        f"""
        <div class="result-banner">
            <strong>Predicted Class:</strong> {result['predicted_class']}<br/>
            <strong>Confidence:</strong> {result['confidence_pct']}<br/>
            <strong>Inference Time:</strong> {result['inference_time_ms']} ms
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(min(max(result["confidence"], 0.0), 1.0))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probability Chart")
        fig = plot_probability_chart(result["probabilities"])
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if overlay is not None:
            st.subheader("Grad-CAM Explanation")
            st.image(overlay, caption="Regions the CNN focused on", use_container_width=True)
        else:
            st.subheader("Tip")
            st.write("Enable Grad-CAM in the options above to visualize important regions.")
        st.markdown("</div>", unsafe_allow_html=True)

    report = build_report_text(result, source)
    st.download_button(
        label="Download Prediction Report",
        data=report,
        file_name=f"prediction_report_{int(time.time())}.txt",
        mime="text/plain",
    )


def main() -> None:
    init_session_state()
    render_header()
    meta = render_sidebar()

    model_ready = MODEL_PATH.exists()
    if not model_ready:
        st.warning(
            "No trained model found at `models/cnn_model.h5`. "
            "Train first with `python train.py`, then refresh this page."
        )

    tab_predict, tab_history, tab_model, tab_charts = st.tabs(
        ["Predict", "History", "Model Summary", "Training Charts"]
    )

    with tab_predict:
        st.markdown("### Upload or Capture an Image")
        opt_cols = st.columns(3)
        with opt_cols[0]:
            show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        with opt_cols[1]:
            use_webcam = st.checkbox("Use Webcam", value=False)

        image_np = None
        source = "upload"

        if use_webcam:
            st.info("Click the button below to capture a frame from your webcam via OpenCV.")
            if st.button("Capture Webcam Frame", disabled=not model_ready):
                try:
                    with st.spinner("Opening webcam..."):
                        st.session_state.webcam_frame = capture_webcam_frame()
                except Exception as exc:
                    logger.exception("Webcam capture failed")
                    st.error(f"Webcam error: {exc}")

            if st.session_state.webcam_frame is not None:
                image_np = st.session_state.webcam_frame
                source = "webcam"
                st.image(image_np, caption="Captured frame", use_container_width=True)
                if st.button("Predict Captured Frame", disabled=not model_ready):
                    run_prediction_ui(image_np, source, show_gradcam)
        else:
            uploaded = st.file_uploader(
                "Drag and drop an image here",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                help="Supports JPG, PNG, BMP, and WebP",
            )
            if uploaded is not None:
                image = Image.open(uploaded).convert("RGB")
                image_np = np.array(image)
                source = f"upload:{uploaded.name}"
                st.image(image, caption="Uploaded image", use_container_width=True)

                if st.button("Predict", disabled=not model_ready, type="primary"):
                    run_prediction_ui(image_np, source, show_gradcam)

    with tab_history:
        st.markdown("### Prediction History")
        if not st.session_state.history:
            st.write("No predictions yet. Classify an image to populate history.")
        else:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download History CSV",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv",
            )
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    with tab_model:
        st.markdown("### Model Architecture Summary")
        if not model_ready:
            st.warning("Train a model to view its summary.")
        else:
            try:
                model = get_cached_model()
                show_model_summary(model)
                st.markdown("#### Training Metadata")
                st.json(meta)
            except Exception as exc:
                st.error(f"Could not load model summary: {exc}")

    with tab_charts:
        st.markdown("### Training Curves & Confusion Matrix")
        c1, c2 = st.columns(2)
        acc_path = IMAGES_DIR / "training_accuracy.png"
        loss_path = IMAGES_DIR / "training_loss.png"
        cm_path = IMAGES_DIR / "confusion_matrix.png"

        with c1:
            if acc_path.exists():
                st.image(str(acc_path), caption="Training Accuracy", use_container_width=True)
            else:
                st.info("training_accuracy.png not found — train the model first.")
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
        with c2:
            if loss_path.exists():
                st.image(str(loss_path), caption="Training Loss", use_container_width=True)
            else:
                st.info("training_loss.png not found — train the model first.")


if __name__ == "__main__":
    main()
