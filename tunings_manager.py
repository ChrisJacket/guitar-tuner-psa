"""
tunings_manager.py
==================
Handles loading, saving, octave inference, and frequency computation for guitar tunings.
"""
import json
import os
import re
from typing import Dict, List, Optional, Tuple

NOTE_ALIASES = {
    "C": 0, "B#": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4, "FB": 4,
    "F": 5, "E#": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11, "CB": 11
}

NOTE_NAMES_STANDARD = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_hz(midi_note: float) -> float:
    """Convert MIDI note number to frequency in Hz (A4 = 440 Hz = MIDI 69)."""
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def hz_to_midi(freq_hz: float) -> float:
    """Convert frequency in Hz to fractional MIDI note number."""
    import numpy as np
    return 69.0 + 12.0 * np.log2(max(freq_hz, 1e-6) / 440.0)


def parse_note_token(token: str) -> Tuple[str, Optional[int]]:
    """
    Parse a token like 'E', 'F#', 'Eb', 'E2', 'C#4' into (canonical_name, octave).
    """
    m = re.match(r"^([A-Ga-g][#bB]?)(-?\d+)?$", token.strip())
    if not m:
        raise ValueError(f"Invalid note token: {token}")
    name_raw = m.group(1).capitalize()
    if len(name_raw) > 1 and name_raw[1].lower() == 'b':
        name_raw = name_raw[0].upper() + 'b'
    elif len(name_raw) > 1 and name_raw[1] == '#':
        name_raw = name_raw[0].upper() + '#'
    else:
        name_raw = name_raw.upper()

    octave = int(m.group(2)) if m.group(2) is not None else None
    return name_raw, octave


def pitch_class_of(name: str) -> int:
    key = name.upper()
    if key not in NOTE_ALIASES:
        raise ValueError(f"Unknown note name: {name}")
    return NOTE_ALIASES[key]


def infer_starting_octave(first_pitch_class: int, num_strings: int) -> int:
    """
    Infer the octave of the lowest string.
    Anchored around standard guitar tuning:
      - 6-string standard lowest note is E2 (pitch class 4).
        If lowest note is C, C#, D, D#, E, F, F#, G, G#, it's octave 2.
        If lowest note is A, A#, B (e.g. Drop A, Drop B), it's octave 1.
      - 7-string standard lowest note is B1 / A1 (octave 1).
      - 8-string standard lowest note is F#1 / E1 (octave 1).
    """
    if num_strings >= 7:
        return 1
    # For 6 or fewer strings:
    # Pitch classes: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8 (octave 2)
    # A=9, A#=10, B=11 are lower than C2 in drop tunings (Drop A1, Drop B1)
    if first_pitch_class in (9, 10, 11):
        return 1
    return 2


def infer_string_notes(notes_str: str, num_strings: Optional[int] = None) -> List[Tuple[str, int, float]]:
    """
    Given a space-separated string of notes (e.g. 'E A D G B E' or 'D2 A2 D3 G3 B3 E4'),
    returns a list of tuples: (display_name_with_octave, midi_number, target_frequency_hz).
    """
    tokens = [t for t in notes_str.strip().split() if t]
    if not tokens:
        return []

    parsed = [parse_note_token(t) for t in tokens]
    n_strings = num_strings if num_strings is not None else len(parsed)

    result: List[Tuple[str, int, float]] = []
    prev_midi = -1

    for i, (name, explicit_octave) in enumerate(parsed):
        pc = pitch_class_of(name)
        if explicit_octave is not None:
            octave = explicit_octave
            midi_num = 12 * (octave + 1) + pc
        else:
            if i == 0:
                octave = infer_starting_octave(pc, n_strings)
                midi_num = 12 * (octave + 1) + pc
            else:
                # Find the smallest octave such that midi_num > prev_midi
                # Normal adjacent guitar string intervals are 1..8 semitones
                octave = (prev_midi - pc) // 12 - 1
                while True:
                    midi_num = 12 * (octave + 1) + pc
                    if midi_num > prev_midi:
                        break
                    octave += 1

        prev_midi = midi_num
        freq = midi_to_hz(midi_num)
        display_name = f"{name}{octave}"
        result.append((display_name, midi_num, freq))

    return result


class TuningsManager:
    """Manages tunings from tunings.json, active tuning state, and persistence."""

    def __init__(self, filepath: str = "tunings.json"):
        self.filepath = filepath
        self.data: dict = {
            "DEFAULT_TUNING": "E standard",
            "LAST_SELECTED_TUNING": "",
            "ALL_TUNINGS": {
                "6-String": {
                    "E standard": "E A D G B E"
                }
            }
        }
        self.active_tuning_name: str = "E standard"
        self.active_category: str = "6-String"
        self.load()

    def load(self) -> None:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Error loading {self.filepath}: {e}")

        # Determine initial active tuning
        last = self.data.get("LAST_SELECTED_TUNING", "").strip()
        default = self.data.get("DEFAULT_TUNING", "E standard").strip()

        target = last if last and self.find_tuning_notes(last) else default
        if not self.find_tuning_notes(target):
            # Fallback to first available tuning
            for cat, tunings in self.get_categories().items():
                if tunings:
                    target = list(tunings.keys())[0]
                    break

        self.set_active_tuning(target)

    def save(self) -> None:
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Error saving {self.filepath}: {e}")

    def get_categories(self) -> Dict[str, Dict[str, str]]:
        return self.data.get("ALL_TUNINGS", {})

    def find_tuning_notes(self, name: str) -> Optional[str]:
        for cat, tunings in self.get_categories().items():
            if name in tunings:
                return tunings[name]
        return None

    def find_category_of(self, name: str) -> Optional[str]:
        for cat, tunings in self.get_categories().items():
            if name in tunings:
                return cat
        return None

    def set_active_tuning(self, name: str) -> bool:
        notes_str = self.find_tuning_notes(name)
        if notes_str is not None:
            self.active_tuning_name = name
            self.active_category = self.find_category_of(name) or "6-String"
            self.data["LAST_SELECTED_TUNING"] = name
            return True
        return False

    def add_custom_tuning(self, category: str, name: str, notes_str: str) -> bool:
        if not name.strip() or not notes_str.strip():
            return False
        # Validate that string notes can be parsed
        try:
            parsed = infer_string_notes(notes_str)
            if not parsed:
                return False
        except Exception:
            return False

        all_tunings = self.data.setdefault("ALL_TUNINGS", {})
        cat_dict = all_tunings.setdefault(category, {})
        cat_dict[name.strip()] = notes_str.strip()
        self.set_active_tuning(name.strip())
        self.save()
        return True

    def get_active_strings(self) -> List[Tuple[str, int, float]]:
        notes_str = self.find_tuning_notes(self.active_tuning_name)
        if not notes_str:
            notes_str = "E A D G B E"
        return infer_string_notes(notes_str)

    def on_exit(self) -> None:
        self.data["LAST_SELECTED_TUNING"] = self.active_tuning_name
        self.save()
