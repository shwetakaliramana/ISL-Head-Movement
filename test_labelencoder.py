"""
Test: does LabelEncoder sort labels alphabetically?
"""
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Test 1: fit with our CLASSES order
classes = ["NOD", "SHAKE", "TILT_LEFT", "TILT_RIGHT", "STATIC"]
le1 = LabelEncoder()
le1.fit(classes)

print("Test 1: LabelEncoder with CLASSES order")
print(f"  Fitted with: {classes}")
print(f"  le.classes_: {le1.classes_}")
for cls in classes:
    idx = le1.transform([cls])[0]
    print(f"    '{cls}' → {idx}")

# Test 2: fit with a pre-sorted order (what if something is sorting it?)
classes_sorted = sorted(classes)
le2 = LabelEncoder()
le2.fit(classes_sorted)

print(f"\nTest 2: LabelEncoder with sorted CLASSES")
print(f"  Fitted with: {classes_sorted}")
print(f"  le.classes_: {le2.classes_}")
for cls in classes_sorted:
    idx = le2.transform([cls])[0]
    print(f"    '{cls}' → {idx}")

# Test 3: Check what happens with the actual label strings from CSV
print(f"\nTest 3: Transform actual label strings from CSV")
labels_from_csv = ["TILT_LEFT"] * 5 + ["TILT_RIGHT"] * 5
le3 = LabelEncoder()
le3.fit(classes)
y = le3.transform(labels_from_csv)
print(f"  CSV labels: {labels_from_csv}")
print(f"  Transformed: {y}")
print(f"  Expected: [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]")
print(f"  Match: {np.array_equal(y, [2, 2, 2, 2, 2, 3, 3, 3, 3, 3])}")
