"""
gui/main_window.py
==================
Main application window for Guitar Tuner:
- Top Tuning Bar with string chips and Tuning Dialog launcher
- Tab 1: Live Tuner (real-time pitch detection + distance display + FFT spectrum plot + device selector + level meter)
- Tab 2: File Input (offline pitch analysis + pitch contour plot)
- Tab 3: Note Replay (Sine, Harmonics, Karplus-Strong, Looping with interval)
"""
import math
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from audio_engine import AudioPlayer, LiveAudioInput
from dsp import (
    analyze_audio_file,
    cents_difference,
    compute_rms,
    compute_spectrum,
    detect_pitch_yin,
    find_closest_string,
)
from gui.meter import Readout
from gui.tuning_dialog import TuningDialog
from tunings_manager import TuningsManager, midi_to_hz, parse_note_token, pitch_class_of

# Semitone range for replay: A1 (MIDI 33) to C6 (MIDI 84)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
REPLAY_NOTES = [f"{_PITCH_CLASSES[m % 12]}{m // 12 - 1}" for m in range(33, 85)]


class MainWindow(tk.Tk):
    CHIP_BG, CHIP_FG = "#e4e4ea", "#222222"
    CHIP_ON_BG, CHIP_ON_FG = "#2ecc71", "#ffffff"

    def __init__(self, tunings_mgr: TuningsManager):
        super().__init__()
        self.tunings_mgr = tunings_mgr

        self.title("Guitar Tuner")
        self.geometry("880x820")
        self.minsize(760, 720)

        # Audio systems
        self.audio_input = LiveAudioInput(buffer_size=8192)
        self.audio_player = AudioPlayer()

        self._chips: List[tk.Label] = []
        self._highlighted_chip: Optional[int] = None
        self._live_running = False
        self._last_detected_time: float = 0.0
        self._last_detected_data: Optional[Tuple[str, float, float, float, int]] = None

        self._build_tuning_bar()
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(4, 10))

        self._build_live_tab()
        self._build_file_tab()
        self._build_replay_tab()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_tuning_display()

    # ================= TOP TUNING BAR =================
    def _build_tuning_bar(self):
        bar = ttk.LabelFrame(self, text="Target Tuning", padding=10)
        bar.pack(fill="x", padx=10, pady=(10, 4))

        top_row = ttk.Frame(bar)
        top_row.pack(fill="x")

        self.tuning_title_lbl = ttk.Label(
            top_row,
            text="",
            font=("Helvetica", 12, "bold")
        )
        self.tuning_title_lbl.pack(side="left")

        ttk.Button(
            top_row,
            text="\u2699 Change Tuning\u2026",
            command=self._open_tuning_dialog
        ).pack(side="right", padx=4)

        self.chip_frame = ttk.Frame(bar)
        self.chip_frame.pack(fill="x", pady=(8, 0))

    def _open_tuning_dialog(self):
        TuningDialog(
            parent=self,
            tunings_mgr=self.tunings_mgr,
            on_applied=lambda name: self.refresh_tuning_display()
        )

    def refresh_tuning_display(self):
        name = self.tunings_mgr.active_tuning_name
        cat = self.tunings_mgr.active_category
        self.tuning_title_lbl.config(text=f"Current: {name}  ({cat})")

        strings = self.tunings_mgr.get_active_strings()
        for w in self.chip_frame.winfo_children():
            w.destroy()

        ttk.Label(self.chip_frame, text="Strings (low \u2192 high):", font=("Helvetica", 10)).pack(side="left", padx=(0, 8))
        self._chips = []
        for text, midi_val, freq_val in strings:
            chip = tk.Label(
                self.chip_frame,
                text=text,
                width=6,
                font=("Helvetica", 11, "bold"),
                bg=self.CHIP_BG,
                fg=self.CHIP_FG,
                padx=6,
                pady=4,
                relief="groove"
            )
            chip.pack(side="left", padx=3)
            self._chips.append(chip)

        self.highlight_string(None)
        self._update_replay_strings()
        self._redraw_live_target_lines()

    def highlight_string(self, index: Optional[int]):
        self._highlighted_chip = index
        for i, chip in enumerate(self._chips):
            is_active = (i == index)
            chip.config(
                bg=self.CHIP_ON_BG if is_active else self.CHIP_BG,
                fg=self.CHIP_ON_FG if is_active else self.CHIP_FG
            )

    # ================= TAB 1: LIVE TUNER =================
    def _build_live_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  Live Tuner  ")

        # Top Device and Control Bar
        top_bar = ttk.Frame(tab)
        top_bar.pack(fill="x", pady=(0, 4))

        self.live_btn = ttk.Button(top_bar, text="Start listening", command=self._toggle_live)
        self.live_btn.pack(side="left")

        # Input device dropdown
        ttk.Label(top_bar, text="Input Device:").pack(side="left", padx=(14, 4))
        self.device_choices = LiveAudioInput.get_input_devices()
        self.device_name_var = tk.StringVar(
            value=self.device_choices[0][1] if self.device_choices else "Default System Input"
        )
        self.device_combo = ttk.Combobox(
            top_bar,
            textvariable=self.device_name_var,
            values=[d[1] for d in self.device_choices],
            state="readonly",
            width=28
        )
        self.device_combo.pack(side="left", padx=4)
        self.device_combo.bind("<<ComboboxSelected>>", self._on_device_changed)

        # Input level / VU meter
        ttk.Label(top_bar, text="Mic Level:").pack(side="left", padx=(12, 4))
        self.level_bar = ttk.Progressbar(top_bar, orient="horizontal", length=80, mode="determinate")
        self.level_bar.pack(side="left")

        self.live_status_lbl = ttk.Label(top_bar, text="Ready.", foreground="#666666")
        self.live_status_lbl.pack(side="left", padx=10)

        # Large Readout & Cents Meter
        self.live_readout = Readout(tab)
        self.live_readout.pack(fill="x", pady=(4, 4))
        self.live_readout.clear("Press \u201cStart listening\u201d and pluck a string")

        # Matplotlib Spectrum
        self.live_fig = Figure(figsize=(6.4, 2.5), dpi=100, facecolor="#1e1e24")
        self.live_fig.subplots_adjust(left=0.08, right=0.96, bottom=0.22, top=0.90)

        self.live_ax = self.live_fig.add_subplot(111)
        self.live_ax.set_facecolor("#121216")
        self.live_ax.set_xlim(40, 700)
        self.live_ax.set_ylim(0.0, 1.05)
        self.live_ax.set_xlabel("Frequency (Hz)", color="#cccccc", fontsize=9)
        self.live_ax.set_ylabel("Normalized Magnitude", color="#cccccc", fontsize=9)
        self.live_ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.live_ax.spines.values():
            spine.set_color("#444444")
        self.live_ax.grid(True, color="#2b2b35", linestyle="--", linewidth=0.6)

        # Plot line elements
        (self.live_line,) = self.live_ax.plot([], [], color="#3498db", linewidth=1.5, label="Spectrum")
        (self.live_peak_marker,) = self.live_ax.plot([], [], "ro", markersize=6, label="Detected Pitch")
        self.live_target_line = self.live_ax.axvline(x=0, color="#2ecc71", linestyle="--", alpha=0.0, linewidth=1.5)
        self.live_detected_line = self.live_ax.axvline(x=0, color="#e74c3c", linestyle=":", alpha=0.0, linewidth=1.5)
        self.live_dist_text = self.live_ax.text(
            0.03, 0.92, "", transform=self.live_ax.transAxes,
            color="#2ecc71", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#1b1b1f", ec="#444", alpha=0.85)
        )
        self.live_string_lines = []

        self.live_canvas = FigureCanvasTkAgg(self.live_fig, master=tab)
        self.live_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _get_selected_device_id(self) -> Optional[int]:
        sel = self.device_name_var.get()
        for dev_id, dev_name in self.device_choices:
            if dev_name == sel:
                return dev_id
        return None

    def _on_device_changed(self, event=None):
        if self._live_running:
            self._start_live()

    def _redraw_live_target_lines(self):
        if not hasattr(self, "live_ax"):
            return
        for line in self.live_string_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.live_string_lines.clear()

        strings = self.tunings_mgr.get_active_strings()
        for label, midi_val, freq_val in strings:
            if 40 <= freq_val <= 700:
                l = self.live_ax.axvline(
                    x=freq_val,
                    color="#555566",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.6
                )
                self.live_string_lines.append(l)
        if hasattr(self, "live_canvas"):
            self.live_canvas.draw_idle()

    def _toggle_live(self):
        if self._live_running:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        dev_id = self._get_selected_device_id()
        ok = self.audio_input.start(dev_id)
        if not ok:
            messagebox.showerror("Audio Error", "Failed to access the audio input device.\nPlease try another device from the dropdown.")
            return
        self._live_running = True
        self._last_detected_data = None
        self._last_detected_time = 0.0
        self.live_btn.config(text="Stop listening")
        self.live_status_lbl.config(text="Listening\u2026 Pluck a guitar string.", foreground="#2ecc71")
        self._live_tick()

    def _stop_live(self):
        self._live_running = False
        self.audio_input.stop()
        self.live_btn.config(text="Start listening")
        self.live_status_lbl.config(text="Stopped.", foreground="#888888")
        self.level_bar["value"] = 0
        self.live_readout.clear("Press \u201cStart listening\u201d and pluck a string")
        self.highlight_string(None)
        # Clear live visuals
        self.live_line.set_data([], [])
        self.live_peak_marker.set_data([], [])
        self.live_target_line.set_alpha(0.0)
        self.live_detected_line.set_alpha(0.0)
        self.live_dist_text.set_text("")
        self.live_canvas.draw_idle()

    def _live_tick(self):
        if not self._live_running:
            return

        buf, sr = self.audio_input.get_latest_buffer()
        strings = self.tunings_mgr.get_active_strings()

        # Update input level meter
        rms = compute_rms(buf)
        level_percent = min(100.0, (rms / 0.05) * 100.0) if rms > 0 else 0.0
        self.level_bar["value"] = level_percent

        # Sensitive pitch detection (min_rms=0.0005)
        detected_hz = detect_pitch_yin(buf, sr, min_freq=40.0, max_freq=1000.0, min_rms=0.0005)

        now = time.time()
        if detected_hz is not None and strings:
            s_idx, s_label, target_hz, cents = find_closest_string(detected_hz, strings)
            hz_diff = detected_hz - target_hz

            # Update live readout with distance in real time
            self.live_readout.show(s_label, detected_hz, target_hz, cents)
            self.highlight_string(s_idx)

            self._last_detected_time = now
            self._last_detected_data = (s_label, detected_hz, target_hz, cents, s_idx)

            # Update lines on spectrum
            if 40 <= target_hz <= 700:
                self.live_target_line.set_xdata([target_hz, target_hz])
                self.live_target_line.set_alpha(0.85)
            if 40 <= detected_hz <= 700:
                self.live_detected_line.set_xdata([detected_hz, detected_hz])
                self.live_detected_line.set_alpha(0.85)

            sign_str = "+" if hz_diff >= 0 else ""
            dist_str = f"{s_label}:  Offset {sign_str}{hz_diff:.2f} Hz  ({cents:+.1f} cents)"
            self.live_dist_text.set_text(dist_str)
            dist_color = "#2ecc71" if abs(cents) <= 5.0 else ("#f39c12" if abs(cents) <= 15.0 else "#e74c3c")
            self.live_dist_text.set_color(dist_color)
        else:
            # If silent or unvoiced: hold the last reading for 1.2s before clearing
            if self._last_detected_data is not None and (now - self._last_detected_time < 1.2):
                pass  # Keep previous stable reading on display
            else:
                self.highlight_string(None)
                self.live_target_line.set_alpha(0.0)
                self.live_detected_line.set_alpha(0.0)
                self.live_dist_text.set_text("")

        # Update FFT spectrum visualization
        freqs, mags = compute_spectrum(buf, sr, n_fft=4096, min_freq=40.0, max_freq=700.0)
        if len(freqs) > 0:
            self.live_line.set_data(freqs, mags)
            if detected_hz is not None and 40 <= detected_hz <= 700:
                closest_bin = np.argmin(np.abs(freqs - detected_hz))
                peak_mag = mags[closest_bin] if closest_bin < len(mags) else 0.5
                self.live_peak_marker.set_data([detected_hz], [peak_mag])
            elif self._last_detected_data is not None and (now - self._last_detected_time < 1.2):
                pass
            else:
                self.live_peak_marker.set_data([], [])
            self.live_canvas.draw_idle()

        # Schedule next tick (~20 Hz update rate)
        self.after(50, self._live_tick)

    # ================= TAB 2: FILE INPUT =================
    def _build_file_tab(self):
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="  File Input  ")

        top_bar = ttk.Frame(tab)
        top_bar.pack(fill="x")

        ttk.Button(top_bar, text="Choose audio file\u2026", command=self._choose_audio_file).pack(side="left")
        self.file_name_lbl = ttk.Label(top_bar, text="No file selected", font=("Helvetica", 10))
        self.file_name_lbl.pack(side="left", padx=12)

        self.file_readout = Readout(tab)
        self.file_readout.pack(fill="x", pady=(6, 4))
        self.file_readout.clear("Choose an audio file (.wav, .mp3, .flac) to analyze pitch")

        # Matplotlib Pitch vs Time
        self.file_fig = Figure(figsize=(6.4, 2.5), dpi=100, facecolor="#1e1e24")
        self.file_fig.subplots_adjust(left=0.08, right=0.96, bottom=0.22, top=0.90)

        self.file_ax = self.file_fig.add_subplot(111)
        self.file_ax.set_facecolor("#121216")
        self.file_ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9)
        self.file_ax.set_ylabel("Pitch (Hz)", color="#cccccc", fontsize=9)
        self.file_ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.file_ax.spines.values():
            spine.set_color("#444444")
        self.file_ax.grid(True, color="#2b2b35", linestyle="--", linewidth=0.6)

        self.file_canvas = FigureCanvasTkAgg(self.file_fig, master=tab)
        self.file_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _choose_audio_file(self):
        path = filedialog.askopenfilename(
            title="Choose Guitar Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self.file_name_lbl.config(text=os.path.basename(path))
        self.file_readout.clear("Analyzing audio...")
        self.update_idletasks()

        strings = self.tunings_mgr.get_active_strings()
        try:
            times, pitches, dominant_pitch, closest_info = analyze_audio_file(
                path,
                target_strings=strings
            )
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze audio file:\n{e}")
            self.file_readout.clear("Error analyzing file.")
            return

        self.file_ax.clear()
        self.file_ax.set_facecolor("#121216")
        self.file_ax.set_xlabel("Time (s)", color="#cccccc", fontsize=9)
        self.file_ax.set_ylabel("Pitch (Hz)", color="#cccccc", fontsize=9)
        self.file_ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.file_ax.spines.values():
            spine.set_color("#444444")
        self.file_ax.grid(True, color="#2b2b35", linestyle="--", linewidth=0.6)

        # Plot pitch trajectory
        valid_mask = ~np.isnan(pitches)
        if np.any(valid_mask) and dominant_pitch is not None and closest_info is not None:
            s_idx, s_label, target_hz, cents = closest_info
            self.file_readout.show(s_label, dominant_pitch, target_hz, cents)
            self.highlight_string(s_idx)

            # Scatter/plot pitch over time
            self.file_ax.plot(times, pitches, color="#3498db", linewidth=1.8, label="Detected Pitch (Hz)")
            self.file_ax.axhline(target_hz, color="#2ecc71", linestyle="--", label=f"Target {s_label} ({target_hz:.2f} Hz)")
            self.file_ax.axhline(dominant_pitch, color="#f39c12", linestyle=":", label=f"Dominant ({dominant_pitch:.2f} Hz)")

            # Dynamic Y limits around detected pitch
            valid_p = pitches[valid_mask]
            y_min = max(30, min(np.min(valid_p) * 0.9, target_hz * 0.9))
            y_max = max(np.max(valid_p) * 1.1, target_hz * 1.1)
            self.file_ax.set_ylim(y_min, y_max)
            self.file_ax.legend(facecolor="#1e1e24", edgecolor="#444444", labelcolor="#dddddd", fontsize=8)
        else:
            self.file_readout.clear("No sustained pitch detected in audio file.")
            self.highlight_string(None)

        self.file_canvas.draw_idle()

    # ================= TAB 3: NOTE REPLAY =================
    def _build_replay_tab(self):
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="  Note Replay  ")

        # Large Note & Frequency display
        self.replay_note_lbl = ttk.Label(tab, text="E2", font=("Helvetica", 54, "bold"), anchor="center")
        self.replay_note_lbl.pack(fill="x", pady=(8, 0))

        self.replay_freq_lbl = ttk.Label(tab, text="82.41 Hz", font=("Helvetica", 12), anchor="center")
        self.replay_freq_lbl.pack(fill="x", pady=(0, 10))

        # Semitone Selector Row
        sel_row = ttk.Frame(tab)
        sel_row.pack(pady=8)

        ttk.Button(sel_row, text="\u25c0 \u22121 Semitone", command=lambda: self._replay_step(-1)).pack(side="left", padx=5)
        self.replay_note_var = tk.StringVar(value="E2")
        self.replay_combo = ttk.Combobox(
            sel_row,
            textvariable=self.replay_note_var,
            values=REPLAY_NOTES,
            state="readonly",
            width=7,
            font=("Helvetica", 11)
        )
        self.replay_combo.pack(side="left", padx=5)
        self.replay_combo.bind("<<ComboboxSelected>>", self._replay_note_changed)
        self.replay_note_var.trace_add("write", lambda *a: self._replay_note_changed())

        ttk.Button(sel_row, text="+1 Semitone \u25b6", command=lambda: self._replay_step(1)).pack(side="left", padx=5)

        # Synthesis & Looping Controls
        opts_box = ttk.LabelFrame(tab, text="Sound & Playback Settings", padding=12)
        opts_box.pack(fill="x", padx=20, pady=12)

        opts_grid = ttk.Frame(opts_box)
        opts_grid.pack()

        # Synthesis Timbre
        ttk.Label(opts_grid, text="Timbre / Waveform:").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.synth_var = tk.StringVar(value="Harmonics")
        self.synth_combo = ttk.Combobox(
            opts_grid,
            textvariable=self.synth_var,
            values=["Harmonics", "Sine", "Plucked String"],
            state="readonly",
            width=16
        )
        self.synth_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        # Looping Checkbox
        self.loop_var = tk.BooleanVar(value=False)
        self.loop_chk = ttk.Checkbutton(opts_grid, text="Loop Repetition", variable=self.loop_var)
        self.loop_chk.grid(row=0, column=2, sticky="w", padx=16, pady=4)

        # Repetition Delay
        ttk.Label(opts_grid, text="Delay between loops (s):").grid(row=0, column=3, sticky="w", padx=6, pady=4)
        self.interval_var = tk.DoubleVar(value=1.0)
        self.interval_spin = ttk.Spinbox(
            opts_grid,
            from_=0.1,
            to=10.0,
            increment=0.5,
            textvariable=self.interval_var,
            width=5
        )
        self.interval_spin.grid(row=0, column=4, sticky="w", padx=6, pady=4)

        # Play / Stop Action Buttons
        act_row = ttk.Frame(tab)
        act_row.pack(pady=8)

        self.play_btn = ttk.Button(act_row, text="\u25b6  Play Tone", command=self._replay_play)
        self.play_btn.pack(side="left", padx=8)

        self.stop_btn = ttk.Button(act_row, text="\u25a0  Stop", command=self._replay_stop)
        self.stop_btn.pack(side="left", padx=8)

        # Open Strings Quick Play Box
        self.replay_string_box = ttk.LabelFrame(
            tab,
            text="Open Strings of Active Tuning (click to select and play)",
            padding=10
        )
        self.replay_string_box.pack(fill="x", padx=20, pady=16)

        self._update_replay_strings()
        self._update_replay_labels("E2")

    def _update_replay_labels(self, note_name: str):
        if getattr(self, "_updating_replay", False):
            return
        self._updating_replay = True
        try:
            name_raw, oct_val = parse_note_token(note_name)
            pc = pitch_class_of(name_raw)
            oct_val = oct_val if oct_val is not None else 2
            midi_num = 12 * (oct_val + 1) + pc
            freq = midi_to_hz(midi_num)
            formatted = f"{name_raw}{oct_val}"
            self.replay_note_lbl.config(text=formatted)
            self.replay_freq_lbl.config(text=f"{freq:.2f} Hz")
            if self.replay_note_var.get() != formatted:
                self.replay_note_var.set(formatted)
        except Exception:
            pass
        finally:
            self._updating_replay = False

    def _replay_step(self, delta: int):
        cur = self.replay_note_var.get()
        if cur not in REPLAY_NOTES:
            cur = "E2"
        idx = REPLAY_NOTES.index(cur) + delta
        idx = max(0, min(len(REPLAY_NOTES) - 1, idx))
        new_note = REPLAY_NOTES[idx]
        self._update_replay_labels(new_note)

    def _replay_note_changed(self, event=None):
        new_note = self.replay_note_var.get()
        self._update_replay_labels(new_note)

    _replay_changed = _replay_note_changed

    def _replay_play(self):
        note_name = self.replay_note_var.get()
        try:
            name_raw, oct_val = parse_note_token(note_name)
            pc = pitch_class_of(name_raw)
            oct_val = oct_val if oct_val is not None else 2
            midi_num = 12 * (oct_val + 1) + pc
            freq = midi_to_hz(midi_num)
        except Exception:
            freq = 82.41

        synth = self.synth_var.get()
        loop = self.loop_var.get()
        try:
            interval = float(self.interval_var.get())
        except Exception:
            interval = 1.0

        self.audio_player.play(freq=freq, synth_type=synth, loop=loop, interval_sec=interval)

    def _replay_stop(self):
        self.audio_player.stop()

    def _update_replay_strings(self):
        if not hasattr(self, "replay_string_box"):
            return
        for w in self.replay_string_box.winfo_children():
            w.destroy()

        strings = self.tunings_mgr.get_active_strings()
        for idx, (label, midi_num, freq_val) in enumerate(strings):
            btn = ttk.Button(
                self.replay_string_box,
                text=f"{label} ({freq_val:.1f} Hz)",
                command=lambda lbl=label: self._on_replay_string_btn_click(lbl)
            )
            btn.pack(side="left", padx=4, pady=4)

    def _on_replay_string_btn_click(self, note_label: str):
        self._update_replay_labels(note_label)
        self._replay_play()

    # ================= LIFECYCLE =================
    def _on_close(self):
        self._stop_live()
        self.audio_player.stop()
        self.tunings_mgr.on_exit()
        self.destroy()
