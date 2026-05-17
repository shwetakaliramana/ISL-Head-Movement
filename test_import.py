import sys
import traceback
sys.path.insert(0, '.')

try:
    import src.classification.gesture_state as gs
    print("Import successful")
    print("Module dict keys:", list(gs.__dict__.keys())[:20])
except Exception as e:
    print("Import failed:")
    traceback.print_exc()
