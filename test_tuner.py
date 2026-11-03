"""
test_tuner.py
=============
Automated test suite verifying:
- Tunings manager & octave inference
- YIN pitch detection, cents deviation, closest string matching
- Audio file offline pitch analysis
- Synthesizer algorithms (Sine, Harmonics, Plucked String)
"""
import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from tunings_manager import TuningsManager, infer_string_notes, midi_to_hz
from dsp import (
    cents_difference,
    find_closest_string,
    detect_pitch_yin,
    compute_spectrum,
    analyze_audio_file,
    compute_rms,
)
from audio_engine import NoteSynthesizer, LiveAudioInput


class TestTuningsManager(unittest.TestCase):
    def test_e_standard_inference(self):
        notes = infer_string_notes("E A D G B E")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["E2", "A2", "D3", "G3", "B3", "E4"])

    def test_drop_d_inference(self):
        notes = infer_string_notes("D A D G B E")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["D2", "A2", "D3", "G3", "B3", "E4"])

    def test_drop_b_inference(self):
        notes = infer_string_notes("B F# B E G# C#")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["B1", "F#2", "B2", "E3", "G#3", "C#4"])

    def test_7string_inference(self):
        notes = infer_string_notes("B E A D G B E")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["B1", "E2", "A2", "D3", "G3", "B3", "E4"])

    def test_8string_inference(self):
        notes = infer_string_notes("F# B E A D G B E")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["F#1", "B1", "E2", "A2", "D3", "G3", "B3", "E4"])

    def test_explicit_octaves(self):
        notes = infer_string_notes("C2 G2 C3 F3 A3 D4")
        labels = [n[0] for n in notes]
        self.assertEqual(labels, ["C2", "G2", "C3", "F3", "A3", "D4"])

    def test_persistence_custom_tuning(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create manager with blank json
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write('{"DEFAULT_TUNING": "E standard", "LAST_SELECTED_TUNING": "", "ALL_TUNINGS": {"6-String": {"E standard": "E A D G B E"}}}')

            tm = TuningsManager(tmp_path)
            self.assertEqual(tm.active_tuning_name, "E standard")

            # Add custom tuning
            ok = tm.add_custom_tuning("6-String", "Open D", "D A D F# A D")
            self.assertTrue(ok)
            self.assertEqual(tm.active_tuning_name, "Open D")

            # Check that it reloads from disk
            tm2 = TuningsManager(tmp_path)
            self.assertEqual(tm2.active_tuning_name, "Open D")
            self.assertIn("Open D", tm2.get_categories()["6-String"])

            # Test exit persistence
            tm2.set_active_tuning("E standard")
            tm2.on_exit()

            tm3 = TuningsManager(tmp_path)
            self.assertEqual(tm3.active_tuning_name, "E standard")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestDSP(unittest.TestCase):
    def test_cents_difference(self):
        # Exactly equal -> 0 cents
        self.assertAlmostEqual(cents_difference(440.0, 440.0), 0.0, places=3)
        # 1 octave up -> +1200 cents
        self.assertAlmostEqual(cents_difference(880.0, 440.0), 1200.0, places=3)
        # 1 semitone up -> +100 cents
        self.assertAlmostEqual(cents_difference(440.0 * (2 ** (1/12)), 440.0), 100.0, places=3)
        # 1 semitone down -> -100 cents
        self.assertAlmostEqual(cents_difference(440.0 * (2 ** (-1/12)), 440.0), -100.0, places=3)

    def test_closest_string_matcher(self):
        strings = infer_string_notes("E A D G B E")
        # Test input 83.0 Hz (near E2 ~ 82.41 Hz)
        idx, label, target, cents = find_closest_string(83.0, strings)
        self.assertEqual(label, "E2")
        self.assertAlmostEqual(target, 82.4068, places=2)
        self.assertGreater(cents, 0)  # Slightly sharp

        # Test input 109.0 Hz (near A2 ~ 110.0 Hz)
        idx, label, target, cents = find_closest_string(109.0, strings)
        self.assertEqual(label, "A2")
        self.assertAlmostEqual(target, 110.0, places=2)
        self.assertLess(cents, 0)  # Slightly flat

    def test_pitch_detection_yin(self):
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # Test frequencies: 82.41 (E2), 146.83 (D3), 329.63 (E4)
        for target_f in [82.41, 146.83, 329.63]:
            # Tone with harmonics and noise
            sig = (np.sin(2 * np.pi * target_f * t) +
                   0.5 * np.sin(2 * np.pi * 2 * target_f * t) +
                   0.005 * np.random.randn(len(t)))
            detected = detect_pitch_yin(sig, sr)
            self.assertIsNotNone(detected)
            cents_err = cents_difference(detected, target_f)
            self.assertLess(abs(cents_err), 1.0)

    def test_silence_rejection(self):
        sr = 44100
        silence = np.zeros(2048, dtype=np.float32)
        detected = detect_pitch_yin(silence, sr)
        self.assertIsNone(detected)

        # Low level noise should also be rejected
        noise = 0.0005 * np.random.randn(2048).astype(np.float32)
        detected_noise = detect_pitch_yin(noise, sr)
        self.assertIsNone(detected_noise)

    def test_spectrum_computation(self):
        sr = 44100
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
        sig = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
        freqs, mags = compute_spectrum(sig, sr, n_fft=4096, min_freq=40.0, max_freq=700.0)
        self.assertGreater(len(freqs), 0)
        self.assertEqual(len(freqs), len(mags))
        # Peak frequency should be near 220 Hz
        peak_freq = freqs[np.argmax(mags)]
        self.assertAlmostEqual(peak_freq, 220.0, delta=12.0)

    def test_offline_file_analysis(self):
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Synthesize an A2 string (~110 Hz)
        audio = (0.7 * np.sin(2 * np.pi * 110.0 * t) +
                 0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        # Apply gentle envelope
        audio *= np.hanning(len(audio))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name

        try:
            sf.write(tmp_wav, audio, sr)
            strings = infer_string_notes("E A D G B E")
            times, pitches, dom_pitch, closest = analyze_audio_file(tmp_wav, strings)

            self.assertGreater(len(times), 0)
            self.assertIsNotNone(dom_pitch)
            self.assertAlmostEqual(dom_pitch, 110.0, delta=1.0)
            self.assertIsNotNone(closest)
            self.assertEqual(closest[1], "A2")
            self.assertAlmostEqual(closest[2], 110.0, places=1)
            self.assertLess(abs(closest[3]), 2.0)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)


class TestAudioEngine(unittest.TestCase):
    def test_synthesizers(self):
        sr = 44100
        sine = NoteSynthesizer.generate_sine(440.0, duration=0.2, sample_rate=sr)
        self.assertEqual(len(sine), int(sr * 0.2))
        self.assertGreater(compute_rms(sine), 0.05)

        harm = NoteSynthesizer.generate_harmonics(440.0, duration=0.2, sample_rate=sr)
        self.assertEqual(len(harm), int(sr * 0.2))
        self.assertGreater(compute_rms(harm), 0.05)

        pluck = NoteSynthesizer.generate_plucked_string(440.0, duration=0.2, sample_rate=sr)
        self.assertEqual(len(pluck), int(sr * 0.2))
        self.assertGreater(compute_rms(pluck), 0.05)


if __name__ == "__main__":
    unittest.main()
