"""
Configuration centrale de l'application Underwater Vision AI.
"""

import os
from pathlib import Path

# ─── Chemins racine ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
RUNS_DIR = ROOT_DIR / "runs"
DATASET_DIR = ROOT_DIR / "dataset"
WEIGHTS_DIR = ROOT_DIR / "runs" / "detect" / "train" / "weights"
UPLOADS_DIR = ROOT_DIR / "uploads"
EXPORTS_DIR = ROOT_DIR / "exports"
HISTORY_FILE = ROOT_DIR / "history.json"

# Créer les dossiers nécessaires
for d in [UPLOADS_DIR, EXPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Modèles disponibles ──────────────────────────────────────────────────────
DEFAULT_PRETRAINED_MODEL = str(ROOT_DIR / "yolov8n.pt")
DEFAULT_CUSTOM_MODEL = str(WEIGHTS_DIR / "best.pt")

# Déterminer le modèle à utiliser par défaut
if (WEIGHTS_DIR / "best.pt").exists():
    DEFAULT_MODEL = DEFAULT_CUSTOM_MODEL
else:
    DEFAULT_MODEL = DEFAULT_PRETRAINED_MODEL

# ─── Classes du dataset (5 espèces de poissons) ──────────────────────────────
FISH_CLASSES = [
    "Acanthopagrus Palmaris",
    "Caranx",
    "Chaetodon Vagabundus",
    "Epinephelus",
    "Gerres",
]

# Couleurs par classe (BGR pour OpenCV, RGB pour affichage)
CLASS_COLORS_RGB = {
    0: (255, 107, 107),   # Acanthopagrus - Rouge corail
    1: (78, 205, 196),    # Caranx       - Turquoise
    2: (255, 230, 109),   # Chaetodon    - Jaune d'or
    3: (107, 91, 149),    # Epinephelus  - Violet
    4: (69, 183, 209),    # Gerres       - Bleu océan
}

CLASS_COLORS_HEX = {
    0: "#FF6B6B",
    1: "#4ECDC4",
    2: "#FFE66D",
    3: "#6B5B95",
    4: "#45B7D1",
}

# ─── Paramètres de détection ──────────────────────────────────────────────────
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_IMG_SIZE = 640

# ─── Paramètres d'entraînement par défaut ────────────────────────────────────
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 0.01
YOLO_MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

# ─── Formats acceptés ─────────────────────────────────────────────────────────
ACCEPTED_IMAGE_FORMATS = ["jpg", "jpeg", "png", "bmp", "webp"]
ACCEPTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]
ACCEPTED_DATASET_FORMATS = ["zip", "yaml"]

# ─── Interface ────────────────────────────────────────────────────────────────
APP_TITLE = "Underwater Vision AI"
APP_ICON = "🐠"
APP_VERSION = "1.0.0"

# ─── ROV/AUV intégration ─────────────────────────────────────────────────────
ROV_API_HOST = os.getenv("ROV_API_HOST", "localhost")
ROV_API_PORT = int(os.getenv("ROV_API_PORT", "8765"))
ROV_STREAM_ENABLED = os.getenv("ROV_STREAM_ENABLED", "false").lower() == "true"
