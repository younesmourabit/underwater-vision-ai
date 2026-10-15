"""
Script de test de détection — à exécuter pour valider l'installation.
Usage : python test_detection.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def test_imports():
    print("=== Test des imports ===")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch : {e}")

    try:
        import ultralytics
        print(f"✅ Ultralytics {ultralytics.__version__}")
    except ImportError as e:
        print(f"❌ Ultralytics : {e}")

    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV : {e}")

    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"❌ Streamlit : {e}")

    try:
        import plotly
        print(f"✅ Plotly {plotly.__version__}")
    except ImportError as e:
        print(f"❌ Plotly : {e}")


def test_model_load():
    print("\n=== Test du chargement du modèle ===")
    from app.detector import load_model
    from app.config import DEFAULT_CUSTOM_MODEL, DEFAULT_PRETRAINED_MODEL

    model_path = DEFAULT_CUSTOM_MODEL if Path(DEFAULT_CUSTOM_MODEL).exists() else DEFAULT_PRETRAINED_MODEL
    print(f"Chargement de : {model_path}")

    model, err = load_model(model_path)
    if model:
        print(f"✅ Modèle chargé : {len(model.names)} classes")
        print(f"   Classes : {list(model.names.values())}")
    else:
        print(f"❌ Erreur : {err}")
    return model


def test_image_detection(model):
    print("\n=== Test de détection sur image ===")
    import numpy as np
    from app.detector import detect_image

    # Test sur image synthétique
    test_img = (ROOT / "test.jpg")
    if test_img.exists():
        import cv2
        img = cv2.imread(str(test_img))
        print(f"Image test chargée : {img.shape}")
    else:
        import numpy as np
        img = np.random.randint(0, 128, (640, 640, 3), dtype=np.uint8)
        print("Image synthétique créée : (640, 640, 3)")

    annotated, detections, inf_ms = detect_image(model, img, conf_threshold=0.1)
    print(f"✅ Détection effectuée en {inf_ms:.1f} ms")
    print(f"   Détections trouvées : {len(detections)}")
    for d in detections:
        print(f"   → {d['class_name']} ({d['confidence']:.1%}) @ {d['bbox']}")

    return annotated, detections


def test_model_info(model):
    print("\n=== Test des informations du modèle ===")
    from app.detector import get_model_info
    info = get_model_info(model)
    print(f"✅ Classes : {info['classes']}")
    print(f"   Nombre : {info['num_classes']}")
    print(f"   Tâche : {info['task']}")


if __name__ == "__main__":
    test_imports()
    model = test_model_load()
    if model:
        test_model_info(model)
        annotated, detections = test_image_detection(model)
        print("\n🎉 Tous les tests passés avec succès !")
        print("Lancez l'application avec : streamlit run streamlit_app.py")
    else:
        print("\n⚠️ Le modèle n'a pas pu être chargé. Vérifiez les logs ci-dessus.")
        sys.exit(1)
