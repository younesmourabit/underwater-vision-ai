"""
Page d'entraînement YOLOv8 - Underwater Vision AI.
"""

import io
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

import streamlit as st

from app.config import (
    DATASET_DIR, RUNS_DIR, UPLOADS_DIR,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LR,
    DEFAULT_IMG_SIZE, YOLO_MODELS,
)
from app.trainer import TrainingSession, validate_dataset_yaml, count_dataset_images
from app.utils import get_device_info, timestamp_str


# Session-level training instance
def _get_training_session() -> TrainingSession:
    if "training_session" not in st.session_state:
        st.session_state["training_session"] = TrainingSession()
    return st.session_state["training_session"]


def render():
    """Render the training page."""

    st.markdown("""
    <div class="page-header">
        <h1>🏋️ Entraînement du Modèle</h1>
        <p class="subtitle">Entraînez YOLOv8 sur votre propre dataset ou continuez avec le dataset existant.</p>
    </div>
    """, unsafe_allow_html=True)

    session = _get_training_session()
    device_info = get_device_info()

    # ─── Info device ─────────────────────────────────────────────────────────
    device_badge = "🟢 GPU" if device_info["cuda_available"] else "🔵 CPU"
    device_name = device_info["device_name"]
    st.info(f"{device_badge} Dispositif de calcul : **{device_name}**")

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    tab_existing, tab_custom = st.tabs(["📂 Dataset existant", "📦 Importer un dataset"])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_existing:
        _render_existing_dataset_tab(session, device_info)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_custom:
        _render_custom_dataset_tab(session, device_info)


def _render_existing_dataset_tab(session: TrainingSession, device_info: dict):
    """Onglet pour entraîner avec le dataset existant."""
    yaml_path = DATASET_DIR / "data.yaml"

    st.markdown("### 📋 Dataset actuel")
    valid, msg, info = validate_dataset_yaml(str(yaml_path))

    if valid:
        st.success(msg)
        # Infos dataset
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <strong>📁 Chemin :</strong> <code>dataset/data.yaml</code><br>
                <strong>🔢 Classes :</strong> {info.get('nc', '?')}<br>
                <strong>📝 Noms :</strong> {', '.join(info.get('names', []))}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            counts = count_dataset_images(str(yaml_path))
            if counts:
                st.markdown("""<div class="info-card">""", unsafe_allow_html=True)
                for split, count in counts.items():
                    icon = {"train": "🏋️", "val": "📊", "test": "🧪"}.get(split, "📄")
                    st.markdown(f"**{icon} {split.capitalize()} :** {count:,} images")
                st.markdown("</div>", unsafe_allow_html=True)

        # Paramètres d'entraînement
        st.markdown("### ⚙️ Configuration de l'entraînement")
        _render_training_form(session, str(yaml_path), device_info, key_prefix="existing")

    else:
        st.error(f"Dataset invalide : {msg}")
        st.info(f"Chemin recherché : `{yaml_path}`")


def _render_custom_dataset_tab(session: TrainingSession, device_info: dict):
    """Onglet pour importer un dataset personnalisé."""
    st.markdown("### 📦 Importer votre dataset")
    st.info("""
    **Format attendu :**
    - ZIP contenant un `data.yaml` à la racine
    - Dossiers `train/images/`, `val/images/` avec leurs annotations YOLO (`.txt`)
    - Le `data.yaml` doit contenir `nc`, `names`, `train`, `val`
    """)

    uploaded_zip = st.file_uploader(
        "Importez votre dataset (ZIP)",
        type=["zip"],
        key="dataset_zip_uploader",
    )

    if uploaded_zip:
        extract_dir = UPLOADS_DIR / "custom_dataset"

        if st.button("📂 Extraire et valider", key="extract_btn"):
            with st.spinner("Extraction en cours..."):
                try:
                    if extract_dir.exists():
                        shutil.rmtree(extract_dir)
                    extract_dir.mkdir(parents=True)

                    with zipfile.ZipFile(io.BytesIO(uploaded_zip.getvalue())) as zf:
                        zf.extractall(extract_dir)
                    st.success("✅ Archive extraite avec succès.")
                    st.session_state["custom_yaml_path"] = str(_find_yaml(extract_dir))

                except Exception as e:
                    st.error(f"Erreur d'extraction : {e}")
                    return

        if "custom_yaml_path" in st.session_state:
            yaml_path = st.session_state["custom_yaml_path"]
            valid, msg, info = validate_dataset_yaml(yaml_path)

            if valid:
                st.success(f"{msg} — {info.get('nc')} classes : {', '.join(info.get('names', []))}")
                counts = count_dataset_images(yaml_path)
                if counts:
                    cols = st.columns(len(counts))
                    for col, (split, count) in zip(cols, counts.items()):
                        col.metric(f"📁 {split}", f"{count:,} imgs")

                st.markdown("### ⚙️ Configuration de l'entraînement")
                _render_training_form(session, yaml_path, device_info, key_prefix="custom")
            else:
                st.error(f"Dataset invalide : {msg}")


def _find_yaml(directory: Path) -> Path:
    """Trouve le premier data.yaml dans une arborescence."""
    for p in directory.rglob("data.yaml"):
        return p
    raise FileNotFoundError("Aucun fichier data.yaml trouvé dans l'archive.")


def _render_training_form(session: TrainingSession, yaml_path: str, device_info: dict, key_prefix: str):
    """Formulaire de configuration et lancement de l'entraînement."""
    col1, col2 = st.columns(2)
    with col1:
        model_name = st.selectbox(
            "🏗️ Modèle YOLOv8",
            YOLO_MODELS,
            index=0,
            help="n=nano, s=small, m=medium, l=large, x=xlarge",
            key=f"{key_prefix}_model",
        )
        epochs = st.number_input("📅 Époques", 1, 500, DEFAULT_EPOCHS, 5, key=f"{key_prefix}_epochs")
        batch_size = st.number_input("📦 Batch size", 1, 64, DEFAULT_BATCH_SIZE, 4, key=f"{key_prefix}_batch")

    with col2:
        img_size = st.selectbox("🖼️ Taille image", [320, 416, 512, 640, 768, 1024], index=3, key=f"{key_prefix}_imgsz")
        lr = st.number_input("📈 Learning rate", 0.0001, 0.1, DEFAULT_LR, 0.001, format="%.4f", key=f"{key_prefix}_lr")
        device = "cuda" if device_info["cuda_available"] else "cpu"
        output_name = f"train_{key_prefix}_{timestamp_str()}"

    st.warning(f"⚠️ Sur CPU, l'entraînement peut prendre plusieurs heures. GPU recommandé pour des résultats rapides.")

    # ─── Boutons de contrôle ─────────────────────────────────────────────────
    col_start, col_stop, _ = st.columns([1, 1, 2])

    with col_start:
        if not session.is_running:
            if st.button("🚀 Démarrer l'entraînement", type="primary", key=f"{key_prefix}_start"):
                ok, msg = session.start(
                    model_name=model_name,
                    data_yaml=yaml_path,
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    lr=lr,
                    output_name=output_name,
                    device=device,
                )
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with col_stop:
        if session.is_running:
            if st.button("⏹️ Arrêter", type="secondary", key=f"{key_prefix}_stop"):
                session.stop()
                st.warning("Arrêt demandé...")

    # ─── Progression ─────────────────────────────────────────────────────────
    if session.is_running:
        st.markdown("### 🔄 Entraînement en cours...")
        logs = session.get_logs()
        if logs:
            log_text = "\n".join(logs)
            st.code(log_text, language="text")
        st.info("L'entraînement tourne en arrière-plan. Rafraîchissez la page pour voir les mises à jour.")
        if st.button("🔄 Rafraîchir", key=f"{key_prefix}_refresh"):
            st.rerun()

    elif session.error:
        st.error(f"❌ Erreur d'entraînement : {session.error}")

    elif session.result is not None:
        st.success("🎉 Entraînement terminé ! Le meilleur modèle a été sauvegardé.")
        best_path = RUNS_DIR / "detect" / output_name / "weights" / "best.pt"
        if best_path.exists():
            with open(best_path, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger best.pt",
                    data=f.read(),
                    file_name=f"best_{output_name}.pt",
                    mime="application/octet-stream",
                )
        session.result = None  # Reset
