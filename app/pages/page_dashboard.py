"""
Page Dashboard & Historique - Underwater Vision AI.
"""

from datetime import datetime
from pathlib import Path

import streamlit as st

from app.config import FISH_CLASSES, CLASS_COLORS_HEX, RUNS_DIR, WEIGHTS_DIR
from app.detector import load_history
from app.utils import get_device_info


def render():
    """Render the dashboard page."""

    st.markdown("""
    <div class="page-header">
        <h1>🌊 Dashboard</h1>
        <p class="subtitle">Vue d'ensemble de votre système de vision sous-marine.</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Statut du système ────────────────────────────────────────────────────
    _render_system_status()

    st.markdown("---")

    # ─── Espèces cibles ───────────────────────────────────────────────────────
    _render_species_info()

    st.markdown("---")

    # ─── Historique des analyses ──────────────────────────────────────────────
    _render_history()

    st.markdown("---")

    # ─── À propos du projet ROV/AUV ───────────────────────────────────────────
    _render_rov_info()


def _render_system_status():
    """Affiche le statut du système en temps réel."""
    st.markdown("### 🖥️ Statut du Système")

    device_info = get_device_info()
    model = st.session_state.get("model")
    model_path = st.session_state.get("model_path", "—")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = "✅ Chargé" if model else "❌ Non chargé"
        color = "#4ECDC4" if model else "#FF6B6B"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {color};">
            <div class="metric-label">🤖 Modèle</div>
            <div class="metric-value" style="color:{color}; font-size:1.1rem;">{status}</div>
            <div style="font-size:0.75rem; opacity:0.6; margin-top:4px;">{Path(model_path).name if model_path != '—' else '—'}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        device_color = "#FFE66D" if device_info["cuda_available"] else "#4ECDC4"
        device_label = "🟡 GPU" if device_info["cuda_available"] else "🔵 CPU"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {device_color};">
            <div class="metric-label">⚡ Dispositif</div>
            <div class="metric-value" style="color:{device_color}; font-size:1rem;">{device_label}</div>
            <div style="font-size:0.75rem; opacity:0.6; margin-top:4px;">{device_info['device_name']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        best_pt = WEIGHTS_DIR / "best.pt"
        weight_exists = best_pt.exists()
        weight_status = "✅ Disponible" if weight_exists else "❌ Absent"
        weight_color = "#4ECDC4" if weight_exists else "#FF6B6B"
        size_str = ""
        if weight_exists:
            size_mb = best_pt.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} Mo"
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid {weight_color};">
            <div class="metric-label">🏋️ Modèle entraîné</div>
            <div class="metric-value" style="color:{weight_color}; font-size:1rem;">{weight_status}</div>
            <div style="font-size:0.75rem; opacity:0.6; margin-top:4px;">{size_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        history = load_history()
        st.markdown(f"""
        <div class="metric-card" style="border-left: 3px solid #6B5B95;">
            <div class="metric-label">📊 Analyses effectuées</div>
            <div class="metric-value" style="color:#6B5B95; font-size:1.5rem;">{len(history)}</div>
            <div style="font-size:0.75rem; opacity:0.6; margin-top:4px;">dans l'historique</div>
        </div>
        """, unsafe_allow_html=True)


def _render_species_info():
    """Affiche les informations sur les espèces détectables."""
    st.markdown("### 🐠 Espèces Détectables")
    st.caption("Dataset : UnderWater Fish Detection (Roboflow) — Licence CC BY 4.0")

    species_info = {
        "Acanthopagrus Palmaris": {
            "desc": "Sparidé des eaux indo-pacifiques, reconnaissable à sa silhouette comprimée.",
            "emoji": "🐟",
            "color": CLASS_COLORS_HEX[0],
        },
        "Caranx": {
            "desc": "Genre de carangues, poissons pélagiques rapides des eaux tropicales.",
            "emoji": "🐠",
            "color": CLASS_COLORS_HEX[1],
        },
        "Chaetodon Vagabundus": {
            "desc": "Poisson-papillon aux rayures caractéristiques, habitant les récifs coralliens.",
            "emoji": "🦋",
            "color": CLASS_COLORS_HEX[2],
        },
        "Epinephelus": {
            "desc": "Mérou, poisson de fond robuste aux taches variables selon l'espèce.",
            "emoji": "🐡",
            "color": CLASS_COLORS_HEX[3],
        },
        "Gerres": {
            "desc": "Ménidés côtiers, petits poissons argentés en bancs près des fonds sableux.",
            "emoji": "🐟",
            "color": CLASS_COLORS_HEX[4],
        },
    }

    cols = st.columns(len(species_info))
    for col, (name, info) in zip(cols, species_info.items()):
        with col:
            st.markdown(f"""
            <div class="species-card" style="border-top: 3px solid {info['color']};">
                <div style="font-size:2rem; margin-bottom:8px;">{info['emoji']}</div>
                <div style="font-weight:700; font-size:0.85rem; color:{info['color']};">{name}</div>
                <div style="font-size:0.75rem; opacity:0.7; margin-top:6px; line-height:1.4;">{info['desc']}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_history():
    """Affiche l'historique des analyses."""
    st.markdown("### 📋 Historique des Analyses")

    history = load_history()

    if not history:
        st.info("Aucune analyse effectuée pour l'instant. Commencez par la page **Détection Image** ou **Vidéo**.")
        return

    # Statistiques globales
    images = [h for h in history if h.get("type") == "image"]
    videos = [h for h in history if h.get("type") == "video"]
    total_dets = sum(h.get("detections", 0) for h in images) + \
                 sum(h.get("total_detections", 0) for h in videos)

    c1, c2, c3 = st.columns(3)
    c1.metric("🖼️ Images analysées", len(images))
    c2.metric("🎬 Vidéos analysées", len(videos))
    c3.metric("🐟 Détections totales", f"{total_dets:,}")

    st.markdown("#### Dernières analyses")

    try:
        import pandas as pd
        rows = []
        for entry in history[:20]:  # Afficher les 20 dernières
            t = entry.get("type", "?")
            icon = "🖼️" if t == "image" else "🎬"
            date_str = entry.get("date", "")
            try:
                dt = datetime.fromisoformat(date_str)
                date_str = dt.strftime("%d/%m/%Y %H:%M")
            except Exception:
                pass

            row = {
                "Type": f"{icon} {t.capitalize()}",
                "Date": date_str,
                "Fichier": entry.get("fichier", "—"),
                "Détections": entry.get("detections") or entry.get("total_detections") or 0,
                "Inférence (ms)": entry.get("inference_ms", "—"),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    except ImportError:
        for entry in history[:10]:
            t = entry.get("type", "?")
            date = entry.get("date", "")[:16]
            file = entry.get("fichier", "—")
            st.text(f"[{t.upper()}] {date} — {file}")

    # Bouton pour vider l'historique
    if st.button("🗑️ Vider l'historique", type="secondary"):
        from app.config import HISTORY_FILE
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.success("Historique vidé.")
        st.rerun()


def _render_rov_info():
    """Section sur l'intégration ROV/AUV future."""
    st.markdown("### 🤖 Intégration ROV/AUV")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#4ECDC4; margin-top:0;">🌊 Vision pour Robot Sous-Marin</h4>
            <p>Ce système est conçu pour une intégration future dans un ROV (Remotely Operated Vehicle) ou AUV (Autonomous Underwater Vehicle).</p>
            <ul>
                <li>🎯 <strong>Détection temps réel</strong> : optimisé YOLOv8n pour l'embarqué</li>
                <li>📡 <strong>API WebSocket</strong> : flux vidéo en temps réel (préparé)</li>
                <li>🗺️ <strong>Cartographie</strong> : localisation des espèces détectées</li>
                <li>🔋 <strong>Embarqué</strong> : compatible Jetson Nano/Orin, Raspberry Pi</li>
            </ul>
            <p style="font-size:0.85rem; opacity:0.7;">Pour activer le mode ROV, définissez <code>ROV_STREAM_ENABLED=true</code> dans les variables d'environnement.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card" style="text-align:center;">
            <div style="font-size:4rem;">🤿</div>
            <div style="font-size:0.9rem; opacity:0.8; margin-top:12px;">
                Prochaines fonctionnalités<br>
                <small>
                    ✦ Flux webcam live<br>
                    ✦ Export ONNX/TensorRT<br>
                    ✦ Carte de chaleur<br>
                    ✦ Alertes automatiques
                </small>
            </div>
        </div>
        """, unsafe_allow_html=True)
