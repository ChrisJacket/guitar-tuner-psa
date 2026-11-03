# Guitar Tuner App Architecture

A modular guitar tuner application in Python built using `numpy`, `scipy`, `matplotlib`, `sounddevice`, and `soundfile`.

---

## 1. Functional Overview

### 1.1 Main Functionality: Live Audio Tuner
- Listens to device's default audio input stream via `sounddevice`.
- Runs real-time pitch detection on windowed audio frames.
- Identifies the closest string note in the currently selected guitar tuning.
- Displays:
  - Detected note name and octave (e.g. `E2`).
  - Detected frequency (Hz) vs target string frequency (Hz).
  - Deviation in cents: $\Delta\text{cents} = 1200 \times \log_2(f_{\text{detected}} / f_{\text{target}})$.
  - Deviation direction and advice (In tune, Too low / tune UP, Too high / tune DOWN).
  - Animated horizontal cents meter with green in-tune zone ($\pm 5$ cents).
  - String indicator chip highlighting the active string.
  - Real-time 1D FFT magnitude spectrum (40–700 Hz) marking detected peak and string frequencies.

### 1.2 Secondary Mode 1: File Input
- Allows the user to select an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.aiff`) using `soundfile`.
- Analyzes the audio signal offline across time frames.
- Determines the dominant pitch and closest tuning string.
- Displays tuning readout and cents deviation.
- Plots pitch contour over time ($f_0$ in Hz vs. Time in seconds) on `matplotlib` canvas.

### 1.3 Secondary Mode 2: Note Replay
- Plays reference tones at half-tone intervals across guitar range ($A_1$ to $C_6$) and open strings of active tuning.
- Provides synthesis options:
  - **Sine Wave**: Pure fundamental frequency.
  - **Harmonics**: Fundamental with decaying acoustic-like harmonic overtones.
  - **Plucked String (Karplus-Strong)**: Physical model simulation of a plucked guitar string.
- Playback controls:
  - **Play** / **Stop** buttons.
  - **Looping** toggle checkbox.
  - **Interval setting**: Configurable delay (in seconds) between repetitions when looping is enabled.
  - Open-string quick play buttons for the current tuning.

### 1.4 Tuning Management
- Reads `tunings.json` at startup:
  - Supports categorized tunings (`6-String`, `7-String`, `8-String`).
  - Restores `LAST_SELECTED_TUNING` if available; otherwise falls back to `DEFAULT_TUNING` (`E standard`).
- **Tuning Selection Window**:
  - Separate modal window opened from the main tuning bar.
  - Contains category tabs (`6-String`, `7-String`, `8-String`, `Custom`).
  - Lists presets in each tab with one-click selection.
  - In the "Custom" tab, allows the user to define a new tuning (name, category, and string note names), automatically saving newly created tunings to `tunings.json`.
- **Octave Inference**:
  - Tuning strings in `tunings.json` store note names without octave numbers (e.g., `E A D G B E`).
  - The app infers appropriate octaves based on guitar register anchored at $E_2$ (low string) and $E_4$ (high string) for 6-string ($B_1$ for 7-string, $F^\sharp_1$ for 8-string), ensuring monotonic or semitone-consistent progression across strings.
- **Persistence**:
  - On app exit, `LAST_SELECTED_TUNING` is persisted to `tunings.json`.
  - Newly created custom tunings are saved directly to `tunings.json`.

---

## 2. System Architecture & Module Structure

The project is structured into clean, decoupled modules with `main.py` serving strictly as the application entry point:

```
├── main.py                 # Application entry point
├── tunings.json            # Tuning presets & persistence file
├── tunings_manager.py      # JSON I/O, octave inference, frequency calculation
├── dsp.py                  # Core signal processing: pitch detection, FFT, cents, file analysis
├── audio_engine.py         # Sounddevice stream capture, synthesizer (Sine, Harmonics, Karplus-Strong), playback thread
├── gui/
│   ├── __init__.py
│   ├── meter.py            # Cents canvas meter and Readout compound widget
│   ├── tuning_dialog.py    # Multi-tab tuning selector and custom tuning creator
│   └── main_window.py      # Main application window (Live, File, Replay tabs)
```

### 2.1 Module Responsibilities

1. **`tunings_manager.py`**:
   - `load_tunings(filepath)` / `save_tunings(filepath, data)`
   - `infer_string_frequencies(notes_str, category_or_num_strings)`: Converts note names (e.g. `"E A D G B E"`) into notes with octaves (`["E2", "A2", "D3", "G3", "B3", "E4"]`) and precise target frequencies in Hz.
   - `TuningsManager` class to track active tuning, available categories, and persistence.

2. **`dsp.py`**:
   - `compute_pitch_autocorr(signal, sample_rate, min_freq=40, max_freq=1000)`: Robust normalized autocorrelation with parabolic interpolation and RMS energy gating.
   - `find_closest_string(detected_hz, string_notes_and_freqs)`: Returns matched string index, note name, target Hz, and cents deviation.
   - `compute_spectrum(signal, sample_rate, min_freq=40, max_freq=700)`: Windowed real FFT magnitude spectrum in dB / amplitude for live visualization.
   - `analyze_audio_file(filepath)`: Framed pitch analysis over time using `soundfile`, returning time array, pitch array, and overall dominant note.

3. **`audio_engine.py`**:
   - `LiveAudioInput`: Manages `sounddevice.InputStream`, ring buffer, thread lock, start/stop.
   - `NoteSynthesizer`: Implements Sine, Harmonic series, and Karplus-Strong synthesis algorithms.
   - `AudioPlayer`: Handles playback queue, looping with configurable interval, and stopping.

4. **`gui/`**:
   - `meter.py`: Reusable canvas-drawn needle meter ($\pm 50$ cents) with green in-tune zone ($\pm 5$ cents) and readout labels.
   - `tuning_dialog.py`: `Toplevel` window with tabs for 6-String, 7-String, 8-String, and Custom. Supports saving custom presets.
   - `main_window.py`: Orchestrates tabs (Live tuner, File input, Note replay) and embeds `matplotlib` figures for spectrum and pitch contour.

5. **`main.py`**:
   - Initializes `TuningsManager`, `AudioPlayer`, `LiveAudioInput`, and `MainWindow`.
   - Binds lifecycle hooks (including saving `LAST_SELECTED_TUNING` on exit).
