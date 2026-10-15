"""
Module de détection YOLOv8 - cœur de la logique de détection.
Gère le chargement du modèle, la détection sur image et vidéo.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.config import (
    DEFAULT_MODEL, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD,
    DEFAULT_IMG_SIZE, CLASS_COLORS_RGB, FISH_CLASSES, HISTORY_FILE,
    EXPORTS_DIR
)

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """
    Charge un modèle YOLOv8. Retourne (model, error_message).
    """
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"Modèle chargé : {model_path}")
        return model, None
    except ImportError:
        return None, "Ultralytics non installé. Lancez : pip install ultralytics"
    except Exception as e:
        return None, f"Erreur lors du chargement du modèle : {str(e)}"


def get_model_info(model) -> dict:
    """Retourne les métadonnées du modèle chargé."""
    try:
        info = {
            "classes": list(model.names.values()) if hasattr(model, "names") else FISH_CLASSES,
            "num_classes": len(model.names) if hasattr(model, "names") else len(FISH_CLASSES),
            "task": model.task if hasattr(model, "task") else "detect",
        }
        return info
    except Exception:
        return {"classes": FISH_CLASSES, "num_classes": len(FISH_CLASSES), "task": "detect"}


def detect_image(
    model,
    image: np.ndarray,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    img_size: int = DEFAULT_IMG_SIZE,
) -> tuple[np.ndarray, list[dict], float]:
    """
    Effectue la détection sur une image NumPy (BGR).

    Retourne:
        annotated_image (np.ndarray): Image annotée avec les boîtes de détection
        detections (list[dict]): Liste des détections avec class, conf, bbox
        inference_time_ms (float): Temps d'inférence en ms
    """
    t_start = time.perf_counter()
    results = model(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        verbose=False,
    )
    inference_time_ms = (time.perf_counter() - t_start) * 1000

    detections = []
    annotated = image.copy()

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                class_name = model.names.get(cls_id, f"Classe {cls_id}")
                color_rgb = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                # Dessiner la boîte
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

                # Fond du label
                label = f"{class_name} {conf:.0%}"
                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color_bgr, -1)
                cv2.putText(
                    annotated, label, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
                )

                detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "width": x2 - x1,
                    "height": y2 - y1,
                })

    return annotated, detections, inference_time_ms


def detect_video(
    model,
    video_path: str,
    output_path: str,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    img_size: int = DEFAULT_IMG_SIZE,
    use_tracker: bool = True,
    progress_callback=None,
) -> dict:
    """
    Effectue la détection sur une vidéo entière.
    Supporte ByteTrack via tracker='bytetrack.yaml'.

    Retourne un dictionnaire de statistiques.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Writer vidéo de sortie
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stats = {
        "total_frames": total_frames,
        "processed_frames": 0,
        "total_detections": 0,
        "class_counts": {},
        "unique_track_ids": set(),
        "inference_times_ms": [],
        "fps": fps,
        "resolution": f"{width}x{height}",
    }

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t_start = time.perf_counter()
            if use_tracker:
                try:
                    results = model.track(
                        frame,
                        conf=conf_threshold,
                        iou=iou_threshold,
                        imgsz=img_size,
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False,
                    )
                except Exception:
                    # Fallback si ByteTrack indisponible
                    results = model(frame, conf=conf_threshold, iou=iou_threshold,
                                   imgsz=img_size, verbose=False)
            else:
                results = model(frame, conf=conf_threshold, iou=iou_threshold,
                               imgsz=img_size, verbose=False)

            inf_ms = (time.perf_counter() - t_start) * 1000
            stats["inference_times_ms"].append(inf_ms)

            annotated = frame.copy()
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf_val = float(box.conf[0].item())
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                        class_name = model.names.get(cls_id, f"Classe {cls_id}")

                        color_rgb = CLASS_COLORS_RGB.get(cls_id, (255, 255, 255))
                        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

                        # Track ID
                        track_id = None
                        if use_tracker and box.id is not None:
                            track_id = int(box.id[0].item())
                            stats["unique_track_ids"].add(track_id)

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

                        label = f"#{track_id} {class_name} {conf_val:.0%}" if track_id else f"{class_name} {conf_val:.0%}"
                        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated, (x1, y1 - lh - bl - 4), (x1 + lw, y1), color_bgr, -1)
                        cv2.putText(annotated, label, (x1, y1 - bl - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        stats["total_detections"] += 1
                        stats["class_counts"][class_name] = stats["class_counts"].get(class_name, 0) + 1

            # Overlay info sur chaque frame
            _draw_video_overlay(annotated, frame_idx, total_frames, stats)
            out.write(annotated)

            frame_idx += 1
            stats["processed_frames"] = frame_idx

            if progress_callback:
                progress_callback(frame_idx, total_frames)

    finally:
        cap.release()
        out.release()

    stats["unique_track_ids"] = len(stats["unique_track_ids"])
    stats["avg_inference_ms"] = (
        np.mean(stats["inference_times_ms"]) if stats["inference_times_ms"] else 0.0
    )
    stats["avg_fps"] = 1000.0 / stats["avg_inference_ms"] if stats["avg_inference_ms"] > 0 else 0.0

    return stats


def _draw_video_overlay(frame: np.ndarray, frame_idx: int, total: int, stats: dict):
    """Dessine un overlay d'information sur une frame vidéo."""
    h, w = frame.shape[:2]
    # Bandeau semi-transparent en haut
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 36), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    progress = f"Frame {frame_idx + 1}/{total}"
    detections = f"Détections: {stats['total_detections']}"
    cv2.putText(frame, progress, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, detections, (w // 2 - 60, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1, cv2.LINE_AA)


def save_to_history(entry: dict):
    """Sauvegarde une entrée dans l'historique JSON."""
    history = load_history()
    history.insert(0, entry)
    history = history[:100]  # Garder max 100 entrées
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Impossible de sauvegarder l'historique : {e}")


def load_history() -> list:
    """Charge l'historique depuis le fichier JSON."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def annotate_summary_stats(detections: list) -> dict:
    """Calcule des statistiques résumées à partir des détections."""
    if not detections:
        return {"total": 0, "by_class": {}, "avg_confidence": 0.0}

    by_class = {}
    for det in detections:
        name = det["class_name"]
        by_class[name] = by_class.get(name, 0) + 1

    avg_conf = np.mean([d["confidence"] for d in detections])
    return {
        "total": len(detections),
        "by_class": by_class,
        "avg_confidence": float(avg_conf),
    }
