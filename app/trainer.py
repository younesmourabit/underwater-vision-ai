"""
Module d'entraînement YOLOv8.
Lance un entraînement sur dataset personnalisé ou existant.
"""

import os
import logging
import threading
import queue
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class TrainingSession:
    """Gère une session d'entraînement avec suivi de progression."""

    def __init__(self):
        self.is_running = False
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = 0
        self.log_queue = queue.Queue()
        self.result = None
        self.error = None
        self._thread: Optional[threading.Thread] = None

    def start(
        self,
        model_name: str,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        img_size: int,
        lr: float,
        output_name: str = "train_custom",
        device: str = "cpu",
    ):
        """Lance l'entraînement dans un thread séparé."""
        if self.is_running:
            return False, "Un entraînement est déjà en cours."

        self.is_running = True
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = epochs
        self.result = None
        self.error = None

        self._thread = threading.Thread(
            target=self._train_worker,
            args=(model_name, data_yaml, epochs, batch_size, img_size, lr, output_name, device),
            daemon=True,
        )
        self._thread.start()
        return True, "Entraînement démarré."

    def _train_worker(self, model_name, data_yaml, epochs, batch_size, img_size, lr, output_name, device):
        try:
            from ultralytics import YOLO

            self.log_queue.put(f"📦 Chargement du modèle {model_name}...")
            model = YOLO(model_name)

            self.log_queue.put(f"🚀 Démarrage de l'entraînement ({epochs} époques)...")

            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=lr,
                name=output_name,
                device=device,
                workers=0,      # Nécessaire sur Windows
                verbose=True,
                plots=True,
                save=True,
            )

            self.result = results
            self.log_queue.put("✅ Entraînement terminé avec succès !")
            self.progress = 1.0

        except Exception as e:
            self.error = str(e)
            self.log_queue.put(f"❌ Erreur : {str(e)}")
        finally:
            self.is_running = False

    def get_logs(self) -> list[str]:
        """Récupère tous les logs disponibles."""
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    def stop(self):
        """Arrête l'entraînement (si possible)."""
        self.is_running = False


def validate_dataset_yaml(yaml_path: str) -> tuple[bool, str, dict]:
    """
    Valide un fichier data.yaml YOLOv8.
    Retourne (valid, message, info).
    """
    try:
        import yaml
    except ImportError:
        try:
            import pyyaml as yaml
        except ImportError:
            return False, "PyYAML non installé.", {}

    path = Path(yaml_path)
    if not path.exists():
        return False, f"Fichier non trouvé : {yaml_path}", {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return False, f"Erreur YAML : {e}", {}

    required = ["nc", "names"]
    for key in required:
        if key not in data:
            return False, f"Clé manquante dans YAML : '{key}'", {}

    nc = data.get("nc", 0)
    names = data.get("names", [])
    if len(names) != nc:
        return False, f"nc={nc} mais {len(names)} classes listées.", {}

    info = {
        "nc": nc,
        "names": names,
        "train": data.get("train", "N/A"),
        "val": data.get("val", "N/A"),
        "test": data.get("test", "N/A"),
    }
    return True, "Dataset YAML valide ✅", info


def count_dataset_images(yaml_path: str) -> dict:
    """Compte les images dans chaque split du dataset."""
    try:
        import yaml as _yaml
    except ImportError:
        return {}

    path = Path(yaml_path)
    parent = path.parent

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _yaml.safe_load(f)
    except Exception:
        return {}

    counts = {}
    for split in ["train", "val", "test"]:
        split_path_str = data.get(split, "")
        if not split_path_str:
            continue

        split_path = Path(split_path_str)
        if not split_path.is_absolute():
            # Essayer relatif au yaml
            split_path = parent / split_path_str
            if not split_path.exists():
                # Essayer relatif au parent du yaml
                split_path = parent.parent / split_path_str.lstrip("../")

        if split_path.exists():
            imgs = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png")) + \
                   list(split_path.glob("*.jpeg")) + list(split_path.glob("*.bmp"))
            counts[split] = len(imgs)
        else:
            counts[split] = 0

    return counts
