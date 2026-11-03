"""
audio_engine.py
===============
Audio hardware input streaming, audio note synthesis (Sine, Harmonics, Karplus-Strong),
and playback management with looping support.
"""
import threading
import time
from typing import List, Optional, Tuple
import numpy as np
import sounddevice as sd


class LiveAudioInput:
    """Manages audio capture from sounddevice.InputStream into a thread-safe ring buffer."""

    def __init__(self, buffer_size: int = 8192):
        self.buffer_size = buffer_size
        self.sample_rate = 44100
        self.device_index: Optional[int] = None
        self.ring_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.lock = threading.Lock()
        self.stream: Optional[sd.InputStream] = None
        self._is_active = False

    @staticmethod
    def get_input_devices() -> List[Tuple[Optional[int], str]]:
        """Return list of (device_index, display_name) for all available input devices."""
        devices: List[Tuple[Optional[int], str]] = [(None, "Default System Input")]
        try:
            dev_list = sd.query_devices()
            for idx, dev in enumerate(dev_list):
                if dev.get("max_input_channels", 0) > 0:
                    api_id = dev.get("hostapi", 0)
                    api_name = sd.query_hostapis(api_id).get("name", "Unknown API")
                    name = f"{dev['name']} ({api_name})"
                    devices.append((idx, name))
        except Exception as e:
            print(f"Error querying devices: {e}")
        return devices

    def _audio_callback(self, indata, frames, time_info, status):
        # Average across channels if stereo / multi-channel
        if indata.ndim > 1 and indata.shape[1] > 1:
            data = np.mean(indata, axis=1)
        elif indata.ndim > 1:
            data = indata[:, 0]
        else:
            data = indata

        n_samples = len(data)
        with self.lock:
            if n_samples >= self.buffer_size:
                self.ring_buffer[:] = data[-self.buffer_size:]
            else:
                self.ring_buffer = np.roll(self.ring_buffer, -n_samples)
                self.ring_buffer[-n_samples:] = data

    def start(self, device_idx: Optional[int] = None) -> bool:
        if self._is_active:
            self.stop()

        self.device_index = device_idx

        # Candidate devices to try if default fails
        devices_to_try = [device_idx] if device_idx is not None else [None]
        if device_idx is None:
            # Add other functional input devices as fallbacks
            try:
                for idx, dev in enumerate(sd.query_devices()):
                    if dev.get("max_input_channels", 0) > 0:
                        devices_to_try.append(idx)
            except Exception:
                pass

        for dev in devices_to_try:
            try:
                if dev is not None:
                    info = sd.query_devices(dev)
                    sr = int(info.get("default_samplerate", 44100))
                    ch = min(2, max(1, info.get("max_input_channels", 1)))
                else:
                    info = sd.query_devices(kind="input")
                    sr = int(info.get("default_samplerate", 44100))
                    ch = min(2, max(1, info.get("max_input_channels", 1)))

                if sr <= 0:
                    sr = 44100
                self.sample_rate = sr

                with self.lock:
                    self.ring_buffer.fill(0.0)

                self.stream = sd.InputStream(
                    device=dev,
                    samplerate=self.sample_rate,
                    channels=ch,
                    dtype="float32",
                    blocksize=1024,
                    callback=self._audio_callback
                )
                self.stream.start()
                self._is_active = True
                self.device_index = dev
                return True
            except Exception as e:
                # Try next candidate device
                if self.stream is not None:
                    try:
                        self.stream.close()
                    except Exception:
                        pass
                    self.stream = None
                continue

        self._is_active = False
        return False

    def stop(self) -> None:
        self._is_active = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
            finally:
                self.stream = None

    def is_running(self) -> bool:
        return self._is_active and (self.stream is not None and self.stream.active)

    def get_latest_buffer(self) -> Tuple[np.ndarray, int]:
        with self.lock:
            buf = self.ring_buffer.copy()
        return buf, self.sample_rate


class NoteSynthesizer:
    """Generates audio samples for notes across various timbres."""

    @staticmethod
    def generate_sine(freq: float, duration: float = 1.5, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = 0.5 * np.sin(2 * np.pi * freq * t)

        # Smooth envelope (attack: 20ms, release: 100ms)
        attack_len = min(int(sample_rate * 0.02), len(waveform) // 4)
        release_len = min(int(sample_rate * 0.1), len(waveform) // 4)

        env = np.ones_like(waveform)
        if attack_len > 0:
            env[:attack_len] = np.linspace(0, 1, attack_len)
        if release_len > 0:
            env[-release_len:] = np.linspace(1, 0, release_len)

        return (waveform * env).astype(np.float32)

    @staticmethod
    def generate_harmonics(freq: float, duration: float = 1.8, sample_rate: int = 44100) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        waveform = np.zeros_like(t)

        # Fundamental + 5 harmonics with physical decay
        harmonics = [
            (1.0, 0.50, 1.0),
            (2.0, 0.30, 1.4),
            (3.0, 0.18, 1.9),
            (4.0, 0.10, 2.5),
            (5.0, 0.05, 3.2),
            (6.0, 0.02, 4.0),
        ]
        nyquist = sample_rate / 2.0
        for mult, amp, decay_rate in harmonics:
            f_h = freq * mult
            if f_h < nyquist:
                decay_env = np.exp(-decay_rate * t)
                waveform += amp * np.sin(2 * np.pi * f_h * t) * decay_env

        # Normalize and apply subtle fade-out at the end
        max_val = np.max(np.abs(waveform))
        if max_val > 1e-5:
            waveform = (waveform / max_val) * 0.6

        release_len = min(int(sample_rate * 0.05), len(waveform) // 4)
        if release_len > 0:
            waveform[-release_len:] *= np.linspace(1, 0, release_len)

        return waveform.astype(np.float32)

    @staticmethod
    def generate_plucked_string(freq: float, duration: float = 2.0, sample_rate: int = 44100) -> np.ndarray:
        """Karplus-Strong physical modeling of a plucked string."""
        period = int(round(sample_rate / freq))
        if period < 2:
            return NoteSynthesizer.generate_harmonics(freq, duration, sample_rate)

        total_samples = int(sample_rate * duration)
        out = np.zeros(total_samples, dtype=np.float32)

        # Initial burst of noise (the pluck)
        ring_buffer = np.random.uniform(-1.0, 1.0, period).astype(np.float32)

        decay_factor = 0.994  # Slight loss factor per cycle
        prev_sample = 0.0

        for i in range(total_samples):
            idx = i % period
            val = ring_buffer[idx]
            # Low-pass averaging filter simulating string stiffness and damping
            filtered = 0.5 * (val + prev_sample) * decay_factor
            ring_buffer[idx] = filtered
            out[i] = val
            prev_sample = val

        # Normalize and fade out tail
        max_val = np.max(np.abs(out))
        if max_val > 1e-5:
            out = (out / max_val) * 0.65

        fade_len = min(int(sample_rate * 0.05), len(out) // 4)
        if fade_len > 0:
            out[-fade_len:] *= np.linspace(1, 0, fade_len)

        return out.astype(np.float32)


class AudioPlayer:
    """Manages audio playback thread, looping, and delay between repetitions."""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_playing = False

    def is_playing(self) -> bool:
        return self._is_playing

    def play(
        self,
        freq: float,
        synth_type: str = "Harmonics",
        loop: bool = False,
        interval_sec: float = 1.0
    ) -> None:
        self.stop()
        self._stop_event.clear()

        def _worker():
            self._is_playing = True
            try:
                while not self._stop_event.is_set():
                    # Synthesize note
                    if synth_type == "Sine":
                        audio = NoteSynthesizer.generate_sine(freq, duration=1.5, sample_rate=self.sample_rate)
                    elif synth_type == "Plucked String":
                        audio = NoteSynthesizer.generate_plucked_string(freq, duration=2.0, sample_rate=self.sample_rate)
                    else:  # "Harmonics" (default)
                        audio = NoteSynthesizer.generate_harmonics(freq, duration=1.8, sample_rate=self.sample_rate)

                    sd.play(audio, samplerate=self.sample_rate)
                    # Wait for playback to complete or stop requested
                    duration = len(audio) / self.sample_rate
                    t_end = time.time() + duration
                    while time.time() < t_end:
                        if self._stop_event.is_set():
                            sd.stop()
                            return
                        time.sleep(0.02)

                    if not loop or self._stop_event.is_set():
                        break

                    # Wait for interval duration between repetitions
                    t_interval_end = time.time() + max(0.0, interval_sec)
                    while time.time() < t_interval_end:
                        if self._stop_event.is_set():
                            break
                        time.sleep(0.02)
            finally:
                self._is_playing = False

        self._thread = threading.Thread(target=_worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.2)
        self._is_playing = False
