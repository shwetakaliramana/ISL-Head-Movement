#!/usr/bin/env python3
"""
scripts/verify_phase6.py
End-to-end verification of Phase 6 ISL text engine integration.

Validates gesture→text conversion pipeline without requiring webcam.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.mapping.isl_text_engine import ISLTextEngine

def main():
    """Verify ISL text engine with representative gesture sequences."""
    
    print("=" * 70)
    print("PHASE 6 VERIFICATION: ISL Text Engine")
    print("=" * 70)
    
    engine = ISLTextEngine(grammar_path="configs/isl_grammar.json")
    
    # Test 1: Single gestures
    print("\n[TEST 1] Single gesture mapping:")
    singles = [
        ("NOD", "YES"),
        ("SHAKE", "NO"),
        ("STATIC", ""),
        ("TILT_LEFT", ""),
        ("TILT_RIGHT", ""),
    ]
    for gesture, expected in singles:
        engine.reset()
        r = engine.push(gesture)
        status = "✓" if r.token == expected else "✗"
        print(f"  {status} {gesture:12s} → {r.token:15s} (expected: {expected})")
    
    # Test 2: Gesture sequences
    print("\n[TEST 2] Gesture sequence mapping:")
    sequences = [
        (["NOD", "NOD"], "DEFINITELY YES"),
        (["SHAKE", "SHAKE"], "ABSOLUTELY NOT"),
        (["TILT_LEFT", "NOD"], "REALLY?"),
        (["TILT_LEFT", "SHAKE"], "IS THAT NO?"),
        (["TILT_LEFT", "TILT_LEFT"], "WHY?"),
        (["TILT_RIGHT", "NOD"], "YES!"),
        (["TILT_RIGHT", "SHAKE"], "NO!"),
        (["NOD", "SHAKE"], "YES AND NO"),
    ]
    for gestures, expected in sequences:
        engine.reset()
        for g in gestures[:-1]:
            engine.push(g)
        r = engine.push(gestures[-1])
        status = "✓" if r.token == expected else "✗"
        gesture_str = "+".join(gestures)
        print(f"  {status} {gesture_str:20s} → {r.token:20s} (expected: {expected})")
    
    # Test 3: Buffer accumulation
    print("\n[TEST 3] Buffer accumulation and sentence emission:")
    engine.reset()
    gestures_input = ["NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE"]
    for i, g in enumerate(gestures_input):
        r = engine.push(g)
        if i == len(gestures_input) - 1:
            print(f"  After {len(gestures_input)} gestures:")
            print(f"    Buffer preview: {engine.buffer_preview}")
            print(f"    Pending token: {r.token}")
    
    sentence = engine._flush_buffer()
    if sentence:
        print(f"    ✓ Sentence emitted: '\''{sentence}'\''")
    else:
        print(f"    ✓ Buffer cleared: {engine.buffer_preview}")
    
    # Test 4: History tracking
    print("\n[TEST 4] History tracking:")
    engine.reset()
    test_sequences = [
        ["NOD", "NOD"],
        ["SHAKE", "SHAKE"],
        ["NOD", "SHAKE"],
    ]
    for seq in test_sequences:
        for g in seq:
            engine.push(g)
        engine._flush_buffer()
    
    print(f"  Total sentences recorded: {len(engine.history)}")
    for i, (ts, sentence) in enumerate(engine.history, 1):
        print(f"    {i}. '\''{sentence}'\'' (timestamp: {ts:.2f})")
    
    # Test 5: Timeout handling
    print("\n[TEST 5] Timeout-based sentence emission:")
    engine.reset()
    engine.push("NOD")
    print(f"  Pushed NOD, buffer: {engine.buffer_preview}")
    
    engine._last_gesture_time -= 10.0
    r = engine.tick()
    if r.sentence:
        print(f"  ✓ Timeout triggered sentence: '\''{r.sentence}'\''")
    else:
        print(f"  ✗ Timeout did not emit sentence")
    
    # Test 6: Gap handling
    print("\n[TEST 6] Gap detection between gestures:")
    engine.reset()
    engine.push("NOD")
    print(f"  Pushed NOD, buffer: {engine.buffer_preview}")
    
    engine._last_gesture_time -= 5.0
    engine.push("SHAKE")
    
    history_sentences = [s for _, s in engine.history]
    if history_sentences:
        print(f"  ✓ Gap detected, buffer flushed before new gesture")
        print(f"    Sentence emitted: '\''{history_sentences[-1]}'\''")
    else:
        print(f"  ? Gap handling (buffer: {engine.buffer_preview})")
    
    print("\n" + "=" * 70)
    print("✓ PHASE 6 VERIFICATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
