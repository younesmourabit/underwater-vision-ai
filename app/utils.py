"""
Utilitaires partagés pour l'application Underwater Vision AI.
"""

import io
import base64
import logging
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convertit une image PIL en ndarray BGR (format OpenCV)."""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convertit un ndarray BGR en image PIL RGB."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def numpy_to_bytes(arr: np.ndarray, format: str = "PNG") -> bytes:
    """Encode un ndarray BGR en bytes PNG/JPEG."""
    pil_img = numpy_to_pil(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()


def format_bytes(size_bytes: int) -> str:
    """Formate une taille en octets de façon lisible."""
    for unit in ["o", "Ko", "Mo", "Go"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} To"


def timestamp_str() -> str:
    """Retourne un timestamp formaté pour les noms de fichiers."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    """Nettoie un nom de fichier pour qu'il soit valide sur Windows."""
    import re
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name[:200]


def get_device_info() -> dict:
    """Retourne les informations sur le dispositif de calcul disponible."""
    info = {"device": "cpu", "cuda_available": False, "device_name": "CPU"}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        pass
    return info


def validate_uploaded_file(
    file_obj,
    accepted_formats: list[str],
    max_size_mb: float = 500,
) -> tuple[bool, str]:
    """
    Valide un fichier uploadé.
    Retourne (valid, error_message).
    """
    if file_obj is None:
        return False, "Aucun fichier sélectionné."

    name = file_obj.name.lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext not in accepted_formats:
        return False, f"Format non supporté : .{ext}. Formats acceptés : {', '.join('.' + f for f in accepted_formats)}"

    size_mb = len(file_obj.getvalue()) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Fichier trop volumineux : {size_mb:.1f} Mo (max {max_size_mb} Mo)."

    return True, ""


def save_uploaded_file(file_obj, destination: Path) -> Path:
    """Sauvegarde un fichier uploadé Streamlit sur disque."""
    dest_path = destination / file_obj.name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(file_obj.getvalue())
    return dest_path


def resize_image_for_display(arr: np.ndarray, max_width: int = 800) -> np.ndarray:
    """Redimensionne une image pour l'affichage si nécessaire."""
    h, w = arr.shape[:2]
    if w <= max_width:
        return arr
    scale = max_width / w
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def create_detection_summary_image(
    class_counts: dict,
    colors_hex: dict,
    class_names: list,
) -> Image.Image:
    """Crée une image de résumé des détections avec des barres colorées."""
    if not class_counts:
        return None

    try:
        import plotly.graph_objects as go

        labels = list(class_counts.keys())
        values = list(class_counts.values())
        colors = [
            colors_hex.get(class_names.index(l), "#888") if l in class_names else "#888"
            for l in labels
        ]

        fig = go.Figure(go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=values,
            textposition="auto",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=max(100, len(labels) * 45),
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2a2a4a"),
            yaxis=dict(gridcolor="#2a2a4a"),
        )
        return fig
    except ImportError:
        return None
