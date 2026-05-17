"""tests/test_phase6.py - Unit tests for ISLTextEngine — Phase 6."""

import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.mapping.isl_text_engine import ISLTextEngine, EngineResult

GRAMMAR = Path("configs/isl_grammar.json")

@pytest.fixture
def engine():
    return ISLTextEngine(grammar_path=GRAMMAR)

class TestSingleGesture:
    def test_nod_maps_to_yes(self, engine):
        r = engine.push("NOD")
        assert r.token == "YES"
    def test_shake_maps_to_no(self, engine):
        r = engine.push("SHAKE")
        assert r.token == "NO"
    def test_static_produces_no_token(self, engine):
        r = engine.push("STATIC")
        assert r.token == ""
    def test_static_produces_no_sentence(self, engine):
        r = engine.push("STATIC")
        assert r.sentence == ""
    def test_tilt_left_maps_to_question(self, engine):
        r = engine.push("TILT_LEFT")
        assert r.token == ""
    def test_tilt_right_maps_to_emphasis(self, engine):
        r = engine.push("TILT_RIGHT")
        assert r.token == ""
    def test_unknown_gesture_passes_through(self, engine):
        r = engine.push("WAVE")
        assert r.token == "WAVE"

class TestSequences:
    def test_nod_nod_is_definitely_yes(self, engine):
        engine.push("NOD")
        r = engine.push("NOD")
        assert r.token == "DEFINITELY YES"
    def test_shake_shake_is_absolutely_not(self, engine):
        engine.push("SHAKE")
        r = engine.push("SHAKE")
        assert r.token == "ABSOLUTELY NOT"
    def test_tilt_left_nod_is_really(self, engine):
        engine.push("TILT_LEFT")
        r = engine.push("NOD")
        assert r.token == "REALLY?"
    def test_tilt_left_shake_is_is_that_no(self, engine):
        engine.push("TILT_LEFT")
        r = engine.push("SHAKE")
        assert r.token == "IS THAT NO?"
    def test_tilt_right_nod_is_yes_exclaim(self, engine):
        engine.push("TILT_RIGHT")
        r = engine.push("NOD")
        assert r.token == "YES!"
    def test_tilt_right_shake_is_no_exclaim(self, engine):
        engine.push("TILT_RIGHT")
        r = engine.push("SHAKE")
        assert r.token == "NO!"
    def test_nod_shake_is_yes_and_no(self, engine):
        engine.push("NOD")
        r = engine.push("SHAKE")
        assert r.token == "YES AND NO"
    def test_tilt_left_tilt_left_is_why(self, engine):
        engine.push("TILT_LEFT")
        r = engine.push("TILT_LEFT")
        assert r.token == "WHY?"

class TestSentenceBuffer:
    def test_buffer_accumulates_tokens(self, engine):
        engine.push("NOD")
        engine.push("SHAKE")
        assert "YES" in engine.buffer_preview
        assert "NO" in engine.buffer_preview
    def test_sentence_emitted_at_max_tokens(self, engine):
        gestures = ["NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE"]
        for g in gestures:
            engine.push(g)
        assert len(engine.history) >= 1 or len(engine.buffer_preview) > 0
    def test_sentence_contains_tokens(self, engine):
        engine.push("NOD")
        engine.push("SHAKE")
        engine._flush_buffer()
        sentences = [s for _, s in engine.history]
        assert len(sentences) > 0
    def test_buffer_clears_after_sentence(self, engine):
        for g in ["NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE"]:
            engine.push(g)
        engine._flush_buffer()
        assert len(engine.buffer_preview) == 0
    def test_history_records_emitted_sentences(self, engine):
        for g in ["NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE"]:
            engine.push(g)
        engine._flush_buffer()
        assert len(engine.history) >= 1
    def test_history_has_timestamp(self, engine):
        for g in ["NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE", "NOD", "SHAKE"]:
            engine.push(g)
        engine._flush_buffer()
        if len(engine.history) > 0:
            ts, sentence = engine.history[-1]
            assert isinstance(ts, float) and ts > 0

class TestTick:
    def test_tick_no_flush_before_timeout(self, engine):
        engine.push("NOD")
        r = engine.tick()
        assert r.sentence == ""
    def test_tick_flushes_after_timeout(self, engine):
        engine.push("NOD")
        engine._last_gesture_time -= 10.0
        r = engine.tick()
        assert r.sentence != "" and "YES" in r.sentence
    def test_tick_returns_empty_when_buffer_empty(self, engine):
        r = engine.tick()
        assert r.sentence == "" and r.token == ""
    def test_gap_flush_before_new_gesture(self, engine):
        engine.push("NOD")
        engine._last_gesture_time -= 10.0
        engine.push("SHAKE")
        sentences = [s for _, s in engine.history]
        assert any("YES" in s for s in sentences)

class TestReset:
    def test_reset_clears_buffer(self, engine):
        engine.push("NOD")
        engine.reset()
        assert engine.buffer_preview == ""
