import sys
sys.path.insert(0, '.')

print("[1] Trying import...")
try:
    from src.ml import feature_engineering as fe
    print("[OK] Module imported")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"[2] Checking module contents:")
print(f"  WINDOW: {hasattr(fe, 'WINDOW')}")
print(f"  N_FEAT: {hasattr(fe, 'N_FEAT')}")
print(f"  CLASSES: {hasattr(fe, 'CLASSES')}")
print(f"  sample_to_features: {hasattr(fe, 'sample_to_features')}")
print(f"  DatasetBuilder: {hasattr(fe, 'DatasetBuilder')}")

print(f"[3] Dir of module:")
for item in dir(fe):
    if not item.startswith('_'):
        print(f"  - {item}")
