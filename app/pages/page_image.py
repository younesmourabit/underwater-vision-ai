"""
Page de détection sur image - Underwater Vision AI.
"""

import io
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.config import (
    ACCEPTED_IMAGE_FORMATS, CLASS_COLORS_HEX, EXPORTS_DIR,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_IMG_SIZE,
)
from app.detector import detect_image, annotate_summary_stats, save_to_history
from app.utils import (
    pil_to_numpy, numpy_to_pil, numpy_to_bytes,
    validate_uploaded_file, save_uploaded_file, resize_image_for_display,
    timestamp_str, get_device_info,
)
from app.config import UPLOADS_DIR


def render():
    """Render the image detection page."""

    st.markdown("""
    <div class="page-header">
        <h1>🔍 Détection sur Image</h1>
        <p class="subtitle">Importez une image sous-marine pour détecter et localiser les poissons.</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Vérification du modèle ───────────────────────────────────────────────
    model = st.session_state.get("model")
    if model is None:
        st.error("⚠️ Aucun modèle chargé. Rendez-vous dans la **Barre latérale** pour charger un modèle.")
        return

    # ─── Paramètres dans la sidebar ───────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres de détection")
        conf_threshold = st.slider(
            "Seuil de confiance", 0.01, 1.0, DEFAULT_CONF_THRESHOLD, 0.01,
            help="Détections en dessous de ce seuil sont ignorées."
        )
        iou_threshold = st.slider(
            "Seuil IOU (NMS)", 0.01, 1.0, DEFAULT_IOU_THRESHOLD, 0.01,
            help="Seuil pour la suppression des doublons."
        )
        img_size = st.selectbox("Taille d'image", [320, 416, 512, 640, 768, 1024], index=3)

    # ─── Upload de l'image ────────────────────────────────────────────────────
    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "📁 Sélectionnez une image",
            type=ACCEPTED_IMAGE_FORMATS,
            help="Formats supportés : JPG, JPEG, PNG, BMP, WEBP",
            key="image_uploader",
        )

    # Aperçu rapide
    if uploaded:
        with col_info:
            size_kb = len(uploaded.getvalue()) / 1024
            pil_img = Image.open(io.BytesIO(uploaded.getvalue()))
            st.markdown(f"""
            <div class="info-card">
                <strong>📄 Fichier :</strong> {uploaded.name}<br>
                <strong>📐 Dimensions :</strong> {pil_img.width} × {pil_img.height} px<br>
                <strong>💾 Taille :</strong> {size_kb:.1f} Ko
            </div>
            """, unsafe_allow_html=True)

    # ─── Validation & Détection ───────────────────────────────────────────────
    if uploaded is None:
        st.markdown("""
        <div class="upload-hint">
            <div style="font-size:4rem;">🐟</div>
            <p>Importez une image pour commencer l'analyse</p>
            <p style="font-size:0.85rem; opacity:0.6;">Formats supportés : JPG, PNG, BMP, WEBP</p>
        </div>
        """, unsafe_allow_html=True)
        return

    valid, err = validate_uploaded_file(uploaded, ACCEPTED_IMAGE_FORMATS, max_size_mb=50)
    if not valid:
        st.error(err)
        return

    # Convertir en numpy BGR
    pil_img = Image.open(io.BytesIO(uploaded.getvalue()))
    img_bgr = pil_to_numpy(pil_img)

    # ─── Bouton de détection ──────────────────────────────────────────────────
    col_btn, col_space = st.columns([1, 3])
    with col_btn:
        detect_btn = st.button("🚀 Lancer la détection", type="primary", use_container_width=True)

    if detect_btn or st.session_state.get("last_detection_image") == uploaded.name:
        with st.spinner("🔍 Analyse en cours..."):
            annotated_bgr, detections, inf_ms = detect_image(
                model, img_bgr,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                img_size=img_size,
            )
            st.session_state["last_detection_image"] = uploaded.name
            st.session_state["last_detections"] = detections
            st.session_state["last_annotated"] = annotated_bgr
            st.session_state["last_inf_ms"] = inf_ms

    # ─── Affichage des résultats ──────────────────────────────────────────────
    if "last_annotated" in st.session_state and st.session_state["last_detection_image"] == uploaded.name:
        annotated_bgr = st.session_state["last_annotated"]
        detections = st.session_state["last_detections"]
        inf_ms = st.session_state["last_inf_ms"]

        # Métriques rapides
        stats = annotate_summary_stats(detections)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🐠 Détections", stats["total"])
        m2.metric("📊 Confiance moy.", f"{stats['avg_confidence']:.0%}" if stats["avg_confidence"] else "—")
        m3.metric("⚡ Inférence", f"{inf_ms:.1f} ms")
        m4.metric("🎯 FPS équiv.", f"{1000/inf_ms:.1f}" if inf_ms > 0 else "—")

        st.markdown("---")

        # Affichage côte à côte
        col_orig, col_annot = st.columns(2)
        with col_orig:
            st.markdown("**📷 Image originale**")
            display_orig = resize_image_for_display(img_bgr)
            st.image(numpy_to_pil(display_orig), use_container_width=True)

        with col_annot:
            st.markdown(f"**✅ Image annotée** ({len(detections)} objets)")
            display_annot = resize_image_for_display(annotated_bgr)
            st.image(numpy_to_pil(display_annot), use_container_width=True)

        # Détails des détections
        if detections:
            st.markdown("### 📋 Détails des détections")
            _render_detection_table(detections)

            # Graphique de répartition des classes
            if stats["by_class"]:
                st.markdown("### 📊 Répartition par espèce")
                _render_class_chart(stats["by_class"])

        else:
            st.info("ℹ️ Aucune détection avec les paramètres actuels. Essayez de réduire le seuil de confiance.")

        # ─── Téléchargement ────────────────────────────────────────────────
        st.markdown("### 💾 Télécharger les résultats")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            img_bytes = numpy_to_bytes(annotated_bgr, "PNG")
            st.download_button(
                "⬇️ Image annotée (PNG)",
                data=img_bytes,
                file_name=f"detection_{timestamp_str()}_{uploaded.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                use_container_width=True,
            )

        with col_dl2:
            if detections:
                import json
                report = {
                    "fichier": uploaded.name,
                    "date": datetime.now().isoformat(),
                    "parametres": {
                        "conf_threshold": conf_threshold,
                        "iou_threshold": iou_threshold,
                        "img_size": img_size,
                    },
                    "statistiques": stats,
                    "detections": detections,
                }
                st.download_button(
                    "⬇️ Rapport JSON",
                    data=json.dumps(report, ensure_ascii=False, indent=2),
                    file_name=f"rapport_{timestamp_str()}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # Sauvegarder dans l'historique
        if detect_btn:
            save_to_history({
                "type": "image",
                "date": datetime.now().isoformat(),
                "fichier": uploaded.name,
                "detections": stats["total"],
                "classes": stats["by_class"],
                "inference_ms": round(inf_ms, 2),
                "conf_threshold": conf_threshold,
            })


def _render_detection_table(detections: list):
    """Affiche un tableau stylisé des détections."""
    import pandas as pd

    rows = []
    for i, det in enumerate(detections, 1):
        w = det["bbox"][2] - det["bbox"][0]
        h = det["bbox"][3] - det["bbox"][1]
        rows.append({
            "#": i,
            "Espèce": det["class_name"],
            "Confiance": f"{det['confidence']:.1%}",
            "Position (x1, y1)": f"({det['bbox'][0]}, {det['bbox'][1]})",
            "Taille (l × h)": f"{det['width']} × {det['height']} px",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_class_chart(by_class: dict):
    """Affiche un graphique en barres horizontal des classes détectées."""
    try:
        import plotly.graph_objects as go
        from app.config import CLASS_COLORS_HEX, FISH_CLASSES

        labels = list(by_class.keys())
        values = list(by_class.values())

        # Couleurs par classe
        colors = []
        for l in labels:
            try:
                idx = [c.lower() for c in FISH_CLASSES].index(l.lower())
                colors.append(CLASS_COLORS_HEX.get(idx, "#4ECDC4"))
            except ValueError:
                colors.append("#4ECDC4")

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
            text=values,
            textposition="auto",
            textfont=dict(color="white", size=14, family="Inter"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,30,0.4)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=max(120, len(labels) * 55),
            font=dict(color="#e0e0ff", family="Inter"),
            xaxis=dict(gridcolor="#2a2a5a", title="Nombre de détections"),
            yaxis=dict(gridcolor="#2a2a5a"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(by_class)
