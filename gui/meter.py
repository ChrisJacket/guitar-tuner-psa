"""
gui/meter.py
============
Cents meter Canvas widget and Readout compound widget.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional

IN_TUNE_CENTS = 5.0  # +/- 5 cents shown as green / in-tune


class Meter(tk.Canvas):
    """Horizontal cents deviation meter: -50 to +50 cents."""
    W, H = 580, 110

    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            width=self.W,
            height=self.H,
            bg="#1b1b1f",
            highlightthickness=0,
            **kwargs
        )
        self.draw(None)

    def _x(self, cents: float) -> float:
        # Map -50..+50 cents to canvas X coordinates with padding
        return 35.0 + (cents + 50.0) / 100.0 * (self.W - 70.0)

    def draw(self, cents: Optional[float]) -> None:
        self.delete("all")
        y0, y1 = 40, 72

        # Background track
        self.create_rectangle(self._x(-50), y0, self._x(50), y1, fill="#2a2a30", outline="")
        # In-tune green center zone
        self.create_rectangle(self._x(-IN_TUNE_CENTS), y0, self._x(IN_TUNE_CENTS), y1, fill="#1f5f3a", outline="")

        # Tick marks
        for c in (-50, -25, 0, 25, 50):
            x = self._x(c)
            self.create_line(x, y0 - 6, x, y1 + 6, fill="#888888", width=1)
            label_text = f"{c:+d}" if c != 0 else "0"
            self.create_text(x, y1 + 18, text=label_text, fill="#aaaaaa", font=("Helvetica", 9))

        # Minor ticks at +/-10 and +/-20
        for c in (-40, -30, -20, -10, 10, 20, 30, 40):
            x = self._x(c)
            self.create_line(x, y0 - 3, x, y1 + 3, fill="#555555", width=1)

        # Flat / Sharp indicators
        self.create_text(self._x(-50), 16, anchor="w", text="\u266d flat (low)", fill="#999999", font=("Helvetica", 10))
        self.create_text(self._x(50), 16, anchor="e", text="sharp (high) \u266f", fill="#999999", font=("Helvetica", 10))

        # Needle indicator
        if cents is not None:
            clamped_cents = max(-50.0, min(50.0, cents))
            x = self._x(clamped_cents)
            abs_c = abs(cents)
            if abs_c <= IN_TUNE_CENTS:
                color = "#2ecc71"  # Green
            elif abs_c <= 15.0:
                color = "#f39c12"  # Amber
            else:
                color = "#e74c3c"  # Red

            self.create_line(x, y0 - 14, x, y1 + 14, fill=color, width=5, capstyle="round")
            self.create_oval(x - 4, y0 + 12, x + 4, y0 + 20, fill="#ffffff", outline=color)


class Readout(ttk.Frame):
    """Compound widget presenting large note name, tuning guidance, meter, and exact numbers."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.note_label = ttk.Label(self, text="\u2014", font=("Helvetica", 52, "bold"), anchor="center")
        self.note_label.pack(fill="x", pady=(2, 0))

        self.hint_label = ttk.Label(self, text="", font=("Helvetica", 13), anchor="center")
        self.hint_label.pack(fill="x")

        self.meter = Meter(self)
        self.meter.pack(pady=6)

        self.info_label = ttk.Label(self, text="", font=("Helvetica", 10), anchor="center")
        self.info_label.pack(fill="x", pady=(0, 2))

    def clear(self, message: str = "") -> None:
        self.note_label.config(text="\u2014")
        self.hint_label.config(text=message, foreground="")
        self.info_label.config(text="")
        self.meter.draw(None)

    def show(self, note: str, detected_hz: float, target_hz: float, cents: float) -> None:
        self.note_label.config(text=note)
        abs_c = abs(cents)
        if abs_c <= IN_TUNE_CENTS:
            self.hint_label.config(text="In tune \u2713", foreground="#1e9e55")
        elif cents < 0:
            self.hint_label.config(text=f"Too low \u2014 tune UP ({abs_c:.1f} cents flat)", foreground="#c0392b")
        else:
            self.hint_label.config(text=f"Too high \u2014 tune DOWN ({abs_c:.1f} cents sharp)", foreground="#c0392b")

        self.info_label.config(
            text=f"Detected: {detected_hz:.2f} Hz   \u2022   Target: {target_hz:.2f} Hz   \u2022   Offset: {cents:+.1f} cents"
        )
        self.meter.draw(cents)
