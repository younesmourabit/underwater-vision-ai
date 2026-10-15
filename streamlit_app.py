"""
Application principale Underwater Vision AI — Point d'entrée Streamlit.

Lancement : streamlit run streamlit_app.py
"""

# ─── Patch cv2 headless AVANT tout autre import ──────────────────────────────
import app.cv2_patch  # noqa: F401 - Must be first import

import sys
import logging
from pathlib import Path

import streamlit as st

# ─── Configuration de la page (DOIT être le premier appel Streamlit) ─────────
st.set_page_config(
    page_title="Underwater Vision AI",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "**Underwater Vision AI** — Détection sous-marine par YOLOv8",
    },
)

# ─── Setup logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ─── CSS global ───────────────────────────────────────────────────────────────
def _load_css():
    css_path = Path(__file__).parent / "app" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css()

# ─── Imports des modules de l'app ────────────────────────────────────────────
from app.config import (
    APP_TITLE, APP_ICON, APP_VERSION,
    DEFAULT_MODEL, DEFAULT_CUSTOM_MODEL, DEFAULT_PRETRAINED_MODEL,
    FISH_CLASSES, CLASS_COLORS_HEX,
)
from app.detector import load_model, get_model_info
from app.utils import get_device_info

# ─── Pages ────────────────────────────────────────────────────────────────────
from app.pages import page_dashboard, page_image, page_video, page_train, page_eval

# ─── Navigation ──────────────────────────────────────────────────────────────
PAGES = {
    "🌊 Dashboard": page_dashboard,
    "🔍 Détection Image": page_image,
    "🎬 Détection Vidéo": page_video,
    "🏋️ Entraînement": page_train,
    "📊 Évaluation": page_eval,
}


def _render_sidebar():
    """Affiche la barre latérale avec la navigation et les contrôles globaux."""
    with st.sidebar:
        # Logo / Titre
        st.markdown(f"""
        <div class="sidebar-brand">
            <div style="font-size:2.5rem; line-height:1;">🐠</div>
            <div>
                <div style="font-weight:800; font-size:1.1rem; letter-spacing:0.5px;">Underwater Vision AI</div>
                <div style="font-size:0.7rem; opacity:0.6; letter-spacing:1px;">v{APP_VERSION} · YOLOv8</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation
        st.markdown("### Navigation")
        current_page = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )

        st.markdown("---")

        # ─── Chargement du modèle ──────────────────────────────────────────
        st.markdown("### 🤖 Modèle")

        model_options = {}
        custom_path = Path(DEFAULT_CUSTOM_MODEL)
        pretrained_path = Path(DEFAULT_PRETRAINED_MODEL)

        if custom_path.exists():
            model_options["✅ Modèle entraîné (best.pt)"] = str(custom_path)
        if pretrained_path.exists():
            model_options["📦 YOLOv8n préentraîné (COCO)"] = str(pretrained_path)
        model_options["📁 Autre chemin..."] = "custom"

        selected_model_label = st.selectbox(
            "Sélectionnez un modèle",
            list(model_options.keys()),
            key="model_selector",
        )

        model_path = model_options[selected_model_label]

        if model_path == "custom":
            model_path = st.text_input(
                "Chemin du modèle (.pt)",
                placeholder="ex: runs/detect/train2/weights/best.pt",
                key="custom_model_path",
            )

        if st.button("🔄 Charger le modèle", type="primary", use_container_width=True):
            if model_path and model_path != "custom":
                with st.spinner("Chargement..."):
                    model, err = load_model(model_path)
                    if model:
                        st.session_state["model"] = model
                        st.session_state["model_path"] = model_path
                        st.session_state["model_info"] = get_model_info(model)
                        st.success("✅ Modèle chargé !")
                    else:
                        st.error(f"❌ {err}")
            else:
                st.warning("Veuillez saisir un chemin valide.")

        # Statut du modèle
        if st.session_state.get("model"):
            info = st.session_state.get("model_info", {})
            path_loaded = st.session_state.get("model_path", "—")
            st.markdown(f"""
            <div class="model-status-ok">
                ✅ <strong>Actif</strong><br>
                <small>{Path(path_loaded).name}</small><br>
                <small>{info.get('num_classes', '?')} classes · {info.get('task', 'detect')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="model-status-err">
                ❌ Aucun modèle chargé
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Device info
        device = get_device_info()
        device_icon = "🟡" if device["cuda_available"] else "🔵"
        st.caption(f"{device_icon} {device['device_name']}")
        st.caption(f"🐍 Python {sys.version.split()[0]} · Streamlit {st.__version__}")

    return current_page


def _auto_load_model():
    """Charge automatiquement le modèle par défaut au démarrage."""
    if "model" not in st.session_state or st.session_state["model"] is None:
        if Path(DEFAULT_CUSTOM_MODEL).exists():
            model, err = load_model(DEFAULT_CUSTOM_MODEL)
            if model:
                st.session_state["model"] = model
                st.session_state["model_path"] = DEFAULT_CUSTOM_MODEL
                st.session_state["model_info"] = get_model_info(model)


def main():
    # Chargement automatique du modèle entraîné
    _auto_load_model()

    # Sidebar et navigation
    current_page = _render_sidebar()

    # Rendu de la page sélectionnée
    page_module = PAGES[current_page]
    page_module.render()


if __name__ == "__main__":
    main()
