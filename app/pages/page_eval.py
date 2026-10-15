"""
Page d'évaluation et métriques - Underwater Vision AI.
"""

import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import streamlit as st

from app.config import RUNS_DIR, DATASET_DIR, DEFAULT_IMG_SIZE
from app.metrics import (
    load_results_csv, extract_final_metrics,
    compute_inference_time, run_validation,
    get_training_curves,
)
from app.utils import get_device_info


def render():
    """Render the evaluation/metrics page."""

    st.markdown("""
    <div class="page-header">
        <h1>📊 Évaluation & Métriques</h1>
        <p class="subtitle">Métriques réelles de performance du modèle : précision, rappel, mAP et temps d'inférence.</p>
    </div>
    """, unsafe_allow_html=True)

    model = st.session_state.get("model")
    train_dir = RUNS_DIR / "detect" / "train"

    # ─── Sélection du run d'entraînement ─────────────────────────────────────
    detect_dir = RUNS_DIR / "detect"
    available_runs = []
    if detect_dir.exists():
        available_runs = [d for d in detect_dir.iterdir() if d.is_dir() and (d / "results.csv").exists()]
        available_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    selected_run = None
    if available_runs:
        run_names = [d.name for d in available_runs]
        selected_name = st.selectbox(
            "📁 Sélectionner un run d'entraînement",
            run_names,
            index=0,
            help="Les runs disponibles avec leurs fichiers de résultats.",
        )
        selected_run = detect_dir / selected_name

    # ─── Tabs ─────────────────────────────────────────────────────────────────
    tab_metrics, tab_curves, tab_infer, tab_val, tab_images = st.tabs([
        "📈 Métriques finales",
        "📉 Courbes d'entraînement",
        "⚡ Temps d'inférence",
        "🔬 Validation complète",
        "🖼️ Visualisations",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        _render_final_metrics(selected_run)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_curves:
        _render_training_curves(selected_run)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_infer:
        _render_inference_timing(model)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_val:
        _render_full_validation(model)

    # ══════════════════════════════════════════════════════════════════════════
    with tab_images:
        _render_training_images(selected_run)


def _render_final_metrics(run_dir: Path):
    """Affiche les métriques finales depuis results.csv."""
    st.markdown("### 📈 Métriques de la dernière époque")

    if run_dir is None:
        st.info("Aucun run d'entraînement disponible. Lancez d'abord un entraînement.")
        return

    metrics = load_results_csv(run_dir)
    if not metrics:
        st.warning("Fichier `results.csv` non trouvé dans ce run.")
        return

    final = extract_final_metrics(metrics)
    if not final:
        st.warning("Impossible d'extraire les métriques. Vérifiez le fichier results.csv.")
        return

    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    precision = final.get("precision")
    recall = final.get("recall")
    map50 = final.get("mAP50")
    map50_95 = final.get("mAP50_95")

    col1.metric("🎯 Précision", f"{precision:.3f}" if precision is not None else "—",
                help="Proportion de détections correctes parmi toutes les détections.")
    col2.metric("📡 Rappel", f"{recall:.3f}" if recall is not None else "—",
                help="Proportion d'objets détectés parmi tous les objets réels.")
    col3.metric("📊 mAP@50", f"{map50:.3f}" if map50 is not None else "—",
                help="Mean Average Precision à IoU=0.50.")
    col4.metric("📊 mAP@50-95", f"{map50_95:.3f}" if map50_95 is not None else "—",
                help="Mean Average Precision moyennée de IoU=0.50 à 0.95.")

    # Losses
    st.markdown("---")
    st.markdown("### 📉 Pertes (dernière époque)")
    lcol1, lcol2, lcol3 = st.columns(3)

    box_loss = final.get("val_box_loss") or final.get("box_loss")
    cls_loss = final.get("val_cls_loss") or final.get("cls_loss")
    dfl_loss = final.get("dfl_loss")

    lcol1.metric("📦 Box Loss", f"{box_loss:.4f}" if box_loss is not None else "—")
    lcol2.metric("🏷️ Cls Loss", f"{cls_loss:.4f}" if cls_loss is not None else "—")
    lcol3.metric("🔧 DFL Loss", f"{dfl_loss:.4f}" if dfl_loss is not None else "—")

    # Interprétation
    if map50 is not None:
        _render_performance_badge(map50)


def _render_performance_badge(map50: float):
    """Affiche un badge de performance selon le mAP."""
    if map50 >= 0.85:
        st.success(f"🏆 Excellent modèle ! mAP@50 = {map50:.1%} — Prêt pour la production.")
    elif map50 >= 0.65:
        st.info(f"✅ Bon modèle. mAP@50 = {map50:.1%} — Améliorable avec plus d'époques ou de données.")
    elif map50 >= 0.45:
        st.warning(f"⚠️ Modèle moyen. mAP@50 = {map50:.1%} — Augmentez les données ou les époques.")
    else:
        st.error(f"❌ Performance faible. mAP@50 = {map50:.1%} — Vérifiez le dataset et les hyperparamètres.")


def _render_training_curves(run_dir: Path):
    """Affiche les courbes d'entraînement interactives."""
    st.markdown("### 📉 Courbes d'entraînement")

    if run_dir is None:
        st.info("Aucun run sélectionné.")
        return

    curves = get_training_curves(run_dir)
    if not curves:
        st.warning("Impossible de charger les courbes (results.csv manquant ou format inattendu).")

        # Afficher results.png si disponible
        results_png = run_dir / "results.png"
        if results_png.exists():
            st.image(str(results_png), caption="Courbes d'entraînement YOLOv8", use_container_width=True)
        return

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Regrouper en 2 graphiques : métriques / losses
        metric_keys = ["mAP50", "mAP50-95", "Précision", "Rappel"]
        loss_keys = ["Box Loss (train)", "Cls Loss (train)", "Box Loss (val)", "Cls Loss (val)"]

        colors_metric = ["#4ECDC4", "#45B7D1", "#FF6B6B", "#FFE66D"]
        colors_loss = ["#FF6B6B", "#FFE66D", "#4ECDC4", "#6B5B95"]

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Métriques de performance", "Fonctions de perte"),
            vertical_spacing=0.1,
        )

        for i, key in enumerate(metric_keys):
            if key in curves:
                data = curves[key]
                fig.add_trace(go.Scatter(
                    x=data["x"], y=data["y"],
                    name=key, line=dict(color=colors_metric[i % len(colors_metric)], width=2),
                    mode="lines+markers", marker=dict(size=4),
                ), row=1, col=1)

        for i, key in enumerate(loss_keys):
            if key in curves:
                data = curves[key]
                fig.add_trace(go.Scatter(
                    x=data["x"], y=data["y"],
                    name=key, line=dict(color=colors_loss[i % len(colors_loss)], width=2,
                                        dash="dot" if "val" in key.lower() else "solid"),
                    mode="lines",
                ), row=2, col=1)

        fig.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,30,0.4)",
            font=dict(color="#e0e0ff", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#3a3a6a"),
        )
        for axis in ["xaxis", "xaxis2", "yaxis", "yaxis2"]:
            fig.update_layout(**{axis: dict(gridcolor="#2a2a5a")})

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("Plotly non disponible. Installation : `pip install plotly`")
        results_png = run_dir / "results.png"
        if results_png.exists():
            st.image(str(results_png), use_container_width=True)


def _render_inference_timing(model):
    """Mesure et affiche le temps d'inférence réel."""
    st.markdown("### ⚡ Benchmark d'inférence")

    if model is None:
        st.error("Aucun modèle chargé.")
        return

    n_runs = st.slider("Nombre de passes de mesure", 5, 50, 20, 5)
    img_size = st.selectbox("Taille d'image", [320, 416, 512, 640], index=2, key="bench_imgsz")

    if st.button("▶️ Lancer le benchmark", type="primary"):
        with st.spinner("Mesure du temps d'inférence..."):
            # Image test synthétique
            test_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
            timing = compute_inference_time(model, test_img, n_runs=n_runs)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⚡ Temps moyen", f"{timing['mean_ms']:.1f} ms")
        c2.metric("📊 Écart-type", f"{timing['std_ms']:.1f} ms")
        c3.metric("⬆️ Max", f"{timing['max_ms']:.1f} ms")
        c4.metric("🎬 FPS", f"{timing['fps']:.1f}")

        device_info = get_device_info()
        st.success(f"Benchmark effectué sur **{device_info['device_name']}** avec {n_runs} passes (2 warmup exclus).")

        # Jauge de performance
        fps = timing["fps"]
        if fps >= 30:
            st.success(f"🎉 {fps:.1f} FPS — Temps réel possible !")
        elif fps >= 10:
            st.info(f"📹 {fps:.1f} FPS — Adapté aux analyses différées.")
        else:
            st.warning(f"🐢 {fps:.1f} FPS — Trop lent pour le temps réel. Utilisez un modèle plus petit ou GPU.")


def _render_full_validation(model):
    """Lance une validation complète sur le dataset."""
    st.markdown("### 🔬 Validation sur le dataset de test")

    if model is None:
        st.error("Aucun modèle chargé.")
        return

    yaml_path = str(DATASET_DIR / "data.yaml")
    conf = st.slider("Seuil de confiance (validation)", 0.01, 1.0, 0.25, 0.01, key="val_conf")

    st.warning("⚠️ La validation peut prendre plusieurs minutes sur CPU.")

    if st.button("🔬 Lancer la validation", type="primary"):
        with st.spinner("Validation en cours (peut prendre quelques minutes)..."):
            val_metrics = run_validation(model, yaml_path, img_size=640, conf=conf)

        if "error" in val_metrics:
            st.error(f"Erreur lors de la validation : {val_metrics['error']}")
        else:
            st.success("✅ Validation terminée !")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Précision", f"{val_metrics.get('precision', 0):.3f}" if val_metrics.get('precision') else "—")
            c2.metric("Rappel", f"{val_metrics.get('recall', 0):.3f}" if val_metrics.get('recall') else "—")
            c3.metric("mAP@50", f"{val_metrics.get('mAP50', 0):.3f}" if val_metrics.get('mAP50') else "—")
            c4.metric("mAP@50-95", f"{val_metrics.get('mAP50_95', 0):.3f}" if val_metrics.get('mAP50_95') else "—")

            if "per_class_ap50" in val_metrics:
                st.markdown("#### AP50 par classe")
                per_class = val_metrics["per_class_ap50"]
                try:
                    import plotly.graph_objects as go
                    from app.config import CLASS_COLORS_HEX, FISH_CLASSES

                    labels = list(per_class.keys())
                    values = list(per_class.values())
                    colors = []
                    for l in labels:
                        try:
                            idx = [c.lower() for c in FISH_CLASSES].index(l.lower())
                            colors.append(CLASS_COLORS_HEX.get(idx, "#4ECDC4"))
                        except ValueError:
                            colors.append("#4ECDC4")

                    fig = go.Figure(go.Bar(
                        x=labels, y=values,
                        marker_color=colors,
                        text=[f"{v:.2f}" for v in values],
                        textposition="auto",
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(10,10,30,0.4)",
                        height=280,
                        font=dict(color="#e0e0ff"),
                        yaxis=dict(range=[0, 1], title="AP@50", gridcolor="#2a2a5a"),
                        xaxis=dict(gridcolor="#2a2a5a"),
                        margin=dict(t=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(per_class)


def _render_training_images(run_dir: Path):
    """Affiche les images générées pendant l'entraînement."""
    st.markdown("### 🖼️ Visualisations de l'entraînement")

    if run_dir is None:
        st.info("Aucun run sélectionné.")
        return

    image_files = {
        "Matrice de confusion": run_dir / "confusion_matrix_normalized.png",
        "Courbe F1": run_dir / "BoxF1_curve.png",
        "Courbe PR": run_dir / "BoxPR_curve.png",
        "Courbe P": run_dir / "BoxP_curve.png",
        "Courbe R": run_dir / "BoxR_curve.png",
        "Labels distribution": run_dir / "labels.jpg",
        "Résultats globaux": run_dir / "results.png",
        "Batch prédictions (val)": run_dir / "val_batch0_pred.jpg",
        "Batch labels (val)": run_dir / "val_batch0_labels.jpg",
    }

    available = {name: path for name, path in image_files.items() if path.exists()}

    if not available:
        st.info(f"Aucune image de visualisation trouvée dans `{run_dir}`.")
        return

    # Afficher en grille 2 colonnes
    items = list(available.items())
    for i in range(0, len(items), 2):
        col1, col2 = st.columns(2)
        with col1:
            name, path = items[i]
            st.markdown(f"**{name}**")
            st.image(str(path), use_container_width=True)
        if i + 1 < len(items):
            with col2:
                name, path = items[i + 1]
                st.markdown(f"**{name}**")
                st.image(str(path), use_container_width=True)
