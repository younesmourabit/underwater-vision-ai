"""
Patch pour résoudre le conflit ultralytics + opencv headless.
Ce module DOIT être importé AVANT ultralytics.
"""

import cv2
import sys

# Patch: si cv2.imshow n'existe pas (headless), créer un stub
if not hasattr(cv2, 'imshow'):
    def _imshow_stub(winname, mat):
        pass  # No-op en mode headless
    cv2.imshow = _imshow_stub

if not hasattr(cv2, 'waitKey'):
    cv2.waitKey = lambda delay=0: -1

if not hasattr(cv2, 'destroyAllWindows'):
    cv2.destroyAllWindows = lambda: None

if not hasattr(cv2, 'namedWindow'):
    cv2.namedWindow = lambda *args, **kwargs: None
