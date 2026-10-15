"""
Page de détection sur vidéo avec tracking ByteTrack - Underwater Vision AI.
"""

import io
import os
import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from app.config import (
    ACCEPTED_VIDEO_FORMATS, EXPORTS_DIR, UPLOADS_DIR,
    DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, DEFAULT_IMG_SIZE,
)
from app.detector import detect_video, save_to_history
from app.utils import (
    validate_uploaded_file, save_uploaded_file, timestamp_str,
)


def render():
    """Render the video detection page."""

    st.markdown("""
    <div class="page-header">
        <h1>🎬 Détection sur Vidéo</h1>
        <p class="subtitle">Analysez une vidéo sous-marine, suivez les poissons avec ByteTrack et exportez les résultats.</p>
    </div>
    """, unsafe_allow_html=True)

    model = st.session_state.get("model")
    if model is None:
        st.error("⚠️ Aucun modèle chargé. Rendez-vous dans la **Barre latérale** pour charger un modèle.")
        return

    # ─── Paramètres sidebar ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres vidéo")
        conf_threshold = st.slider("Seuil de confiance", 0.01, 1.0, DEFAULT_CONF_THRESHOLD, 0.01, key="vid_conf")
        iou_threshold = st.slider("Seuil IOU", 0.01, 1.0, DEFAULT_IOU_THRESHOLD, 0.01, key="vid_iou")
        img_size = st.selectbox("Taille d'image", [320, 416, 512, 640, 768], index=3, key="vid_imgsz")
        use_tracker = st.toggle("🎯 Tracking ByteTrack", value=True, help="Active le suivi d'objets entre les frames.")
        st.markdown("---")
        st.info("💡 Le tracking attribue un ID unique à chaque poisson pour les compter précisément.")

    # ─── Upload de la vidéo ───────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "📁 Sélectionnez une vidéo",
        type=ACCEPTED_VIDEO_FORMATS,
        help="Formats supportés : MP4, AVI, MOV, MKV",
        key="video_uploader",
    )

    if uploaded is None:
        _render_video_placeholder()
        return

    # Validation
    valid, err = validate_uploaded_file(uploaded, ACCEPTED_VIDEO_FORMATS, max_size_mb=500)
    if not valid:
        st.error(err)
        return

    # Infos vidéo
    video_path = save_uploaded_file(uploaded, UPLOADS_DIR)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps if fps > 0 else 0
    cap.release()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📹 Résolution", f"{width}×{height}")
    col2.metric("🎞️ Frames", f"{total_frames:,}")
    col3.metric("⏱️ Durée", f"{duration_s:.1f} s")
    col4.metric("🔄 FPS", f"{fps:.0f}")

    st.markdown("---")

    # ─── Lancer le traitement ─────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyze_btn = st.button("🚀 Analyser la vidéo", type="primary", use_container_width=True)

    if analyze_btn:
        output_filename = f"annotated_{timestamp_str()}_{uploaded.name.rsplit('.', 1)[0]}.mp4"
        output_path = str(EXPORTS_DIR / output_filename)

        progress_bar = st.progress(0, text="Initialisation...")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        processed_count = [0]
        last_update = [time.time()]

        def progress_callback(current, total):
            processed_count[0] = current
            now = time.time()
            if now - last_update[0] > 0.5:  # Mise à jour toutes les 0.5s
                pct = current / total if total > 0 else 0
                progress_bar.progress(pct, text=f"Frame {current}/{total} ({pct:.0%})")
                last_update[0] = now

        try:
            t_start = time.time()
            stats = detect_video(
                model,
                str(video_path),
                output_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                img_size=img_size,
                use_tracker=use_tracker,
                progress_callback=progress_callback,
            )
            elapsed = time.time() - t_start

            progress_bar.progress(1.0, text="✅ Traitement terminé !")
            st.session_state["video_stats"] = stats
            st.session_state["video_output_path"] = output_path
            st.session_state["video_elapsed"] = elapsed

            # Sauvegarder dans l'historique
            save_to_history({
                "type": "video",
                "date": datetime.now().isoformat(),
                "fichier": uploaded.name,
                "frames": stats.get("processed_frames", 0),
                "total_detections": stats.get("total_detections", 0),
                "unique_tracks": stats.get("unique_track_ids", 0),
                "inference_ms": round(stats.get("avg_inference_ms", 0), 2),
            })

        except Exception as e:
            st.error(f"❌ Erreur lors du traitement : {str(e)}")
            return

    # ─── Afficher les résultats ───────────────────────────────────────────────
    if "video_stats" in st.session_state:
        stats = st.session_state["video_stats"]
        output_path = st.session_state.get("video_output_path", "")
        elapsed = st.session_state.get("video_elapsed", 0)

        st.markdown("## 📊 Résultats de l'analyse")

        # Métriques clés
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🐠 Détections totales", f"{stats.get('total_detections', 0):,}")
        m2.metric("🏷️ IDs uniques (pistes)", stats.get("unique_track_ids", "N/A") if use_tracker else "Désactivé")
        m3.metric("⚡ Inférence moy.", f"{stats.get('avg_inference_ms', 0):.1f} ms")
        m4.metric("⏱️ Temps total", f"{elapsed:.1f} s")

        # Répartition par classe
        class_counts = stats.get("class_counts", {})
        if class_counts:
            st.markdown("### 🐟 Espèces détectées")
            _render_class_bar_chart(class_counts)

        # Téléchargement de la vidéo annotée
        output_p = Path(output_path)
        if output_p.exists():
            st.markdown("### 💾 Vidéo annotée")
            st.success(f"Vidéo exportée : `{output_p.name}`")
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Télécharger la vidéo annotée",
                    data=f.read(),
                    file_name=output_p.name,
                    mime="video/mp4",
                    use_container_width=True,
                )

            # Aperçu de la première frame annotée
            st.markdown("### 🖼️ Aperçu de la vidéo annotée")
            cap_out = cv2.VideoCapture(output_path)
            ret, preview_frame = cap_out.read()
            cap_out.release()
            if ret:
                from app.utils import numpy_to_pil
                st.image(numpy_to_pil(preview_frame), caption="Première frame annotée", use_container_width=True)


def _render_video_placeholder():
    st.markdown("""
    <div class="upload-hint">
        <div style="font-size:4rem;">🎬</div>
        <p>Importez une vidéo pour commencer l'analyse</p>
        <p style="font-size:0.85rem; opacity:0.6;">Formats supportés : MP4, AVI, MOV, MKV</p>
    </div>
    """, unsafe_allow_html=True)


def _render_class_bar_chart(class_counts: dict):
    """Graphique en barres des classes détectées."""
    try:
        import plotly.graph_objects as go
        from app.config import CLASS_COLORS_HEX, FISH_CLASSES

        labels = list(class_counts.keys())
        values = list(class_counts.values())

        colors = []
        for l in labels:
            try:
                idx = [c.lower() for c in FISH_CLASSES].index(l.lower())
                colors.append(CLASS_COLORS_HEX.get(idx, "#4ECDC4"))
            except ValueError:
                colors.append("#4ECDC4")

        fig = go.Figure(go.Bar(
            x=labels,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=values,
            textposition="auto",
            textfont=dict(color="white", size=14),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,30,0.4)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            font=dict(color="#e0e0ff", family="Inter"),
            xaxis=dict(gridcolor="#2a2a5a"),
            yaxis=dict(gridcolor="#2a2a5a", title="Nombre de détections"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(class_counts)
