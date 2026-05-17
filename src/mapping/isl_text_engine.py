"""
src/mapping/isl_text_engine.py
─────────────────────────────────────────────────────────────────────────────
ISL Head Movement → Text Conversion Engine.

Sits at the top of the pipeline and receives emitted gesture labels from
the FSM (Phase 3) or the LSTM classifier (Phase 5).  Converts the stream
of discrete gestures into readable ISL-derived text.

Responsibilities:
  1. Single-gesture mapping   NOD → "YES", SHAKE → "NO", etc.
  2. Sequence mapping         TILT_LEFT + NOD → "REALLY?", etc.
  3. Modifier application     TILT_LEFT before a gesture = question marker
  4. Sentence buffering       accumulates tokens until a pause or max length
  5. Sentence emission        fires a complete sentence and clears the buffer
  6. History logging          keeps the last N sentences for display

Usage:
    engine = ISLTextEngine()

    # Called once per emitted gesture (from FSM / LSTM)
    result = engine.push("NOD")
    if result.token:
        print("Token:", result.token)       # e.g. "YES"
    if result.sentence:
        print("Sentence:", result.sentence) # e.g. "REALLY? YES"

    # Called every frame to check if a sentence break has timed out
    result = engine.tick()
    if result.sentence:
        print("Timeout sentence:", result.sentence)
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

_GRAMMAR_PATH = Path(__file__).parents[2] / "configs" / "isl_grammar.json"
_MAX_SENTENCE_TOKENS = 8
_HISTORY_LEN         = 20


@dataclass
class EngineResult:
    """Return value from push() and tick()."""
    token:    str = ""       # single gesture token just produced (may be empty for STATIC)
    sentence: str = ""       # complete sentence if one was emitted this call, else ""
    buffer:   str = ""       # current in-progress token buffer as a string


class ISLTextEngine:
    """
    Converts a stream of discrete gesture labels into ISL text.

    Args:
        grammar_path: path to isl_grammar.json (defaults to configs/).
        max_gap_s:    override max_sequence_gap_seconds from grammar.
        break_s:      override sentence_break_seconds from grammar.
    """

    def __init__(
        self,
        grammar_path: str | Path | None = None,
        max_gap_s:  Optional[float] = None,
        break_s:    Optional[float] = None,
    ) -> None:
        path = Path(grammar_path) if grammar_path else _GRAMMAR_PATH
        with open(path) as f:
            g = json.load(f)

        self._single:   dict[str, str]  = g["single_gesture_map"]
        self._sequence: dict[str, str]  = g["sequence_map"]
        self._modifiers: dict           = g["modifier_rules"]
        timing                          = g["timing"]

        self._max_gap_s = max_gap_s or timing["max_sequence_gap_seconds"]
        self._break_s   = break_s   or timing["sentence_break_seconds"]

        # Buffer: list of (token_str, raw_gesture, timestamp)
        self._buffer:   list[tuple[str, str, float]] = []
        self._pending_modifier: Optional[str] = None   # TILT_LEFT / TILT_RIGHT held

        self._last_gesture_time: float = 0.0

        # Output history
        self._history: deque[tuple[float, str]] = deque(maxlen=_HISTORY_LEN)

        log.info("ISLTextEngine ready (gap=%.1fs, break=%.1fs)", self._max_gap_s, self._break_s)

    # ── Public API ─────────────────────────────────────────────────────────────

    def push(self, gesture: str) -> EngineResult:
        """
        Receive a newly emitted gesture label and update internal state.

        Args:
            gesture: one of NOD, SHAKE, TILT_LEFT, TILT_RIGHT, STATIC.
        Returns:
            EngineResult with token, sentence (if emitted), and buffer preview.
        """
        now = time.time()

        # ── Gap check: if too long since last gesture, flush first ────────────
        if (self._buffer and
                now - self._last_gesture_time > self._max_gap_s):
            self._flush_buffer()

        self._last_gesture_time = now

        # ── STATIC: no token, just update timing ──────────────────────────────
        if gesture == "STATIC":
            return EngineResult(buffer=self._buffer_preview())

        sentence_out = ""
        token = ""
        
        # ── Check for sequence BEFORE processing modifiers ────────────────────
        # Sequences can match even if last gesture was a modifier (empty token)
        if self._buffer:
            last_gesture = self._buffer[-1][1]
            seq_key = f"{last_gesture}+{gesture}"
            if seq_key in self._sequence:
                # Found a sequence — emit the sequence token
                token = self._sequence[seq_key]
                log.debug("Sequence match: %s → %s", seq_key, token)
                # Remove the old gesture that matched
                self._buffer.pop()
                # Add the sequence token
                self._buffer.append((token, gesture, now))
                self._pending_modifier = None  # Clear modifier after using in sequence
                
                # Check if buffer is full after adding
                if len(self._buffer) >= _MAX_SENTENCE_TOKENS:
                    sentence_out = self._flush_buffer()
                
                return EngineResult(
                    token=token,
                    sentence=sentence_out,
                    buffer=self._buffer_preview(),
                )

        # ── Modifier gestures (TILT_LEFT / TILT_RIGHT) ───────────────────────
        # Store modifiers in buffer for sequence matching, but don't emit tokens
        mod_rule = self._modifiers.get(gesture, {})
        if mod_rule.get("applies_to_next"):
            self._pending_modifier = gesture
            self._buffer.append(("", gesture, now))  # Store modifier as empty token
            log.debug("Modifier held: %s", gesture)
            return EngineResult(buffer=self._buffer_preview())

        # ── Resolve token ─────────────────────────────────────────────────────
        token = self._resolve_token(gesture)
        
        self._buffer.append((token, gesture, now))
        log.debug("Buffer: %s", [t for t, _, _ in self._buffer])

        # ── Flush if buffer is full ────────────────────────────────────────────
        if len(self._buffer) >= _MAX_SENTENCE_TOKENS:
            sentence_out = self._flush_buffer()

        return EngineResult(
            token=token,
            sentence=sentence_out,
            buffer=self._buffer_preview(),
        )

    def tick(self) -> EngineResult:
        """
        Call every frame (or periodically) to detect sentence-break timeouts.

        Returns EngineResult with sentence set if a timeout flush occurred.
        """
        if not self._buffer:
            return EngineResult()

        elapsed = time.time() - self._last_gesture_time
        if elapsed >= self._break_s:
            sentence = self._flush_buffer()
            return EngineResult(sentence=sentence, buffer="")

        return EngineResult(buffer=self._buffer_preview())

    def reset(self) -> None:
        """Hard reset — clears buffer and pending modifier."""
        self._buffer.clear()
        self._pending_modifier = None
        log.info("ISLTextEngine reset.")

    @property
    def history(self) -> list[tuple[float, str]]:
        """List of (timestamp, sentence_text) for the last N sentences."""
        return list(self._history)

    @property
    def buffer_preview(self) -> str:
        return self._buffer_preview()

    # ── Private helpers ────────────────────────────────────────────────────────

    def _resolve_token(self, gesture: str) -> str:
        """Map gesture → text token, applying any pending modifier."""
        base = self._single.get(gesture, gesture)

        if self._pending_modifier:
            mod = self._pending_modifier
            self._pending_modifier = None
            seq_key = f"{mod}+{gesture}"
            if seq_key in self._sequence:
                return self._sequence[seq_key]
            # No specific sequence rule — prepend modifier symbol
            mod_symbol = self._single.get(mod, "")
            return f"{mod_symbol} {base}".strip() if mod_symbol else base

        return base

    def _flush_buffer(self) -> str:
        """Join buffered tokens into a sentence, add to history, clear buffer."""
        if not self._buffer:
            return ""
        tokens   = [t for t, _, _ in self._buffer]
        sentence = " ".join(t for t in tokens if t)   # skip empty (STATIC)
        self._buffer.clear()
        self._pending_modifier = None
        if sentence:
            self._history.append((time.time(), sentence))
            log.info("Sentence emitted: '%s'", sentence)
        return sentence

    def _buffer_preview(self) -> str:
        tokens = [t for t, _, _ in self._buffer]
        if self._pending_modifier:
            tokens.append(f"[{self._pending_modifier}…]")
        return " ".join(t for t in tokens if t)
