import numpy as np
import sys
sys.path.insert(0, '.')

def make_nod_angles(n=5, amplitude=15.0):
    t = np.arange(n) * 4 * np.pi / n  # Corrected to avoid sine zeros
    pitches = amplitude * np.sin(t)
    return [{"yaw": 0.0, "pitch": float(p), "roll": 0.0} for p in pitches]

angles = make_nod_angles(n=5)
print("Nod angles:")
for i, a in enumerate(angles):
    print(f"  Frame {i}: pitch={a['pitch']:.2f}")

print("\nDeltas:")
for i in range(1, len(angles)):
    delta = abs(angles[i]["pitch"] - angles[i-1]["pitch"])
    print(f"  Delta {i}: {delta:.2f}")

# Try the FSM
from src.classification.gesture_state import GestureFSM, GestureState
fsm = GestureFSM()
print(f"\nInitial state: {fsm.ctx.state}")
for i, a in enumerate(angles):
    result = fsm.update(a["yaw"], a["pitch"], a["roll"])
    print(f"  After frame {i}: state={fsm.ctx.state}, result={result}")
