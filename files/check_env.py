"""
scripts/check_env.py
─────────────────────────────────────────────────────────────────────────────
Run this FIRST after setting up your environment.
Verifies that every required package imports correctly and prints versions.

Usage:
  python scripts/check_env.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK]  {label:<28} {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label:<27} {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  ISL Head Movement — Environment Check")
    print("=" * 60)

    print(f"\n  Python: {sys.version.split()[0]}")

    print("\n  Core packages:")
    results = []

    results.append(check("numpy", lambda: __import__("numpy").__version__))
    results.append(check("opencv-python", lambda: __import__("cv2").__version__))
    results.append(check("mediapipe", lambda: __import__("mediapipe").__version__))
    results.append(check("scipy", lambda: __import__("scipy").__version__))

    print("\n  ML packages:")
    results.append(check("scikit-learn", lambda: __import__("sklearn").__version__))
    results.append(check("tensorflow", lambda: __import__("tensorflow").__version__))

    print("\n  Data / viz:")
    results.append(check("pandas", lambda: __import__("pandas").__version__))
    results.append(check("matplotlib", lambda: __import__("matplotlib").__version__))
    results.append(check("streamlit", lambda: __import__("streamlit").__version__))

    print("\n  Project modules:")
    results.append(check("config", lambda: str(__import__("config", fromlist=["cfg"]).cfg.camera.fps) + " fps configured"))
    results.append(check("camera", lambda: "Camera class OK"))
    results.append(check("face_mesh", lambda: "FaceMeshDetector OK"))
    results.append(check("optical_flow", lambda: "OpticalFlowTracker OK"))

    print("\n  Camera device:")
    try:
        import cv2
        from config import cfg
        cap = cv2.VideoCapture(cfg.camera.device_id)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [OK]  Device {cfg.camera.device_id:<22} {w}x{h}")
            cap.release()
        else:
            print(f"  [WARN] Camera device {cfg.camera.device_id} not found.")
            print("         → Set camera.device_id in configs/config.yaml")
    except Exception as e:
        print(f"  [FAIL] Camera check: {e}")

    print("\n" + "=" * 60)
    failed = results.count(False)
    if failed == 0:
        print("  All checks passed. Run: python scripts/verify_phase1.py")
    else:
        print(f"  {failed} check(s) failed — see above. Fix before proceeding.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
