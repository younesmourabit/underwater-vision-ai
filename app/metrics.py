"""
Module de métriques et d'évaluation YOLOv8.
Lit les fichiers résultats réels produits par l'entraînement.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import RUNS_DIR

logger = logging.getLogger(__name__)


def load_results_csv(train_dir: Path) -> Optional[dict]:
    """
    Charge le fichier results.csv produit par Ultralytics.
    Retourne un dict de listes de valeurs par métrique, ou None si absent.
    """
    csv_path = train_dir / "results.csv"
    if not csv_path.exists():
        return None

    metrics = {}
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    key = key.strip()
                    try:
                        val = float(val.strip())
                    except (ValueError, AttributeError):
                        continue
                    metrics.setdefault(key, []).append(val)
    except Exception as e:
        logger.warning(f"Impossible de lire results.csv : {e}")
        return None

    return metrics


def extract_final_metrics(metrics: dict) -> dict:
    """
    Extrait les métriques finales (dernière époque) depuis le dict de metrics.
    """
    key_map = {
        "precision": ["metrics/precision(B)", "precision"],
        "recall": ["metrics/recall(B)", "recall"],
        "mAP50": ["metrics/mAP50(B)", "mAP50"],
        "mAP50_95": ["metrics/mAP50-95(B)", "mAP50-95"],
        "box_loss": ["train/box_loss", "box_loss"],
        "cls_loss": ["train/cls_loss", "cls_loss"],
        "dfl_loss": ["train/dfl_loss", "dfl_loss"],
        "val_box_loss": ["val/box_loss"],
        "val_cls_loss": ["val/cls_loss"],
    }

    result = {}
    for friendly_name, possible_keys in key_map.items():
        for k in possible_keys:
            if k in metrics and metrics[k]:
                result[friendly_name] = metrics[k][-1]
                break

    return result


def compute_inference_time(model, test_image: np.ndarray, n_runs: int = 20) -> dict:
    """
    Mesure le temps d'inférence réel en effectuant n_runs passes.
    Retourne mean, std, min, max en ms.
    """
    import time
    times = []
    for _ in range(n_runs):
        t = time.perf_counter()
        model(test_image, verbose=False)
        times.append((time.perf_counter() - t) * 1000)

    times = times[2:]  # Enlever le warmup
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "fps": float(1000 / np.mean(times)) if np.mean(times) > 0 else 0.0,
    }


def run_validation(model, data_yaml: str, img_size: int = 640, conf: float = 0.25) -> dict:
    """
    Lance une validation complète avec Ultralytics et retourne les métriques réelles.
    """
    try:
        results = model.val(
            data=data_yaml,
            imgsz=img_size,
            conf=conf,
            verbose=False,
            plots=False,
        )

        box = results.box if hasattr(results, "box") else None
        metrics = {}
        if box:
            metrics["precision"] = float(np.mean(box.p)) if hasattr(box, "p") else None
            metrics["recall"] = float(np.mean(box.r)) if hasattr(box, "r") else None
            metrics["mAP50"] = float(box.map50) if hasattr(box, "map50") else None
            metrics["mAP50_95"] = float(box.map) if hasattr(box, "map") else None
            if hasattr(box, "ap_class_index") and hasattr(box, "ap50"):
                metrics["per_class_ap50"] = {
                    model.names.get(int(i), f"Classe {i}"): float(ap)
                    for i, ap in zip(box.ap_class_index, box.ap50)
                }
        return metrics
    except Exception as e:
        logger.error(f"Erreur de validation : {e}")
        return {"error": str(e)}


def get_training_curves(train_dir: Path) -> dict:
    """
    Charge les courbes d'entraînement depuis results.csv.
    Retourne un dict prêt pour Plotly/matplotlib.
    """
    metrics = load_results_csv(train_dir)
    if not metrics:
        return {}

    curves = {}
    epoch_key = next((k for k in metrics if "epoch" in k.lower()), None)
    epochs = metrics.get(epoch_key, list(range(len(next(iter(metrics.values()))))))

    desired = {
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
        "Précision": "metrics/precision(B)",
        "Rappel": "metrics/recall(B)",
        "Box Loss (train)": "train/box_loss",
        "Cls Loss (train)": "train/cls_loss",
        "Box Loss (val)": "val/box_loss",
        "Cls Loss (val)": "val/cls_loss",
    }

    for label, key in desired.items():
        # Chercher aussi sans espaces / variantes
        found_key = None
        for k in metrics:
            if k.strip() == key.strip():
                found_key = k
                break
            if key.replace("/", "_").lower() in k.replace("/", "_").lower():
                found_key = k
                break
        if found_key:
            curves[label] = {"x": list(epochs), "y": metrics[found_key]}

    return curves
