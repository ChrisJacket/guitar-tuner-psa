"""
dsp.py
======
Digital Signal Processing algorithms for guitar tuner:
- Fundamental frequency (f0) detection using YIN / Autocorrelation
- Parabolic interpolation for sub-sample accuracy
- RMS energy gating with sensitive threshold for microphone input
- DC offset removal and high-pass conditioning
- Cents difference calculation and string matching
- Fast Fourier Transform (FFT) magnitude spectrum
- Audio file offline analysis
"""
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal
import soundfile as sf


def cents_difference(detected_hz: float, target_hz: float) -> float:
    """Calculate pitch error in cents: 1200 * log2(detected_hz / target_hz)."""
    if detected_hz <= 0 or target_hz <= 0:
        return 0.0
    return 1200.0 * float(np.log2(detected_hz / target_hz))


def find_closest_string(
    detected_hz: float,
    strings: List[Tuple[str, int, float]]
) -> Tuple[int, str, float, float]:
    """
    Find the closest string from the active tuning.
    strings: list of (note_label, midi_num, target_hz).
    Returns (string_index, note_label, target_hz, cents_diff).
    """
    if not strings or detected_hz <= 0:
        return 0, "", 0.0, 0.0

    best_idx = 0
    best_cents = float("inf")
    best_target = strings[0][2]
    best_label = strings[0][0]

    for idx, (label, midi_num, target_hz) in enumerate(strings):
        diff = cents_difference(detected_hz, target_hz)
        if abs(diff) < abs(best_cents):
            best_cents = diff
            best_idx = idx
            best_target = target_hz
            best_label = label

    return best_idx, best_label, best_target, best_cents


def parabolic_interpolation(array: np.ndarray, x: int) -> Tuple[float, float]:
    """
    Parabolic interpolation around peak index x.
    Returns (refined_x, interpolated_value).
    """
    if x <= 0 or x >= len(array) - 1:
        return float(x), float(array[x])

    alpha = float(array[x - 1])
    beta = float(array[x])
    gamma = float(array[x + 1])

    denom = alpha - 2.0 * beta + gamma
    if abs(denom) < 1e-12:
        return float(x), beta

    delta = 0.5 * (alpha - gamma) / denom
    refined_x = x + delta
    val = beta - 0.25 * (alpha - gamma) * delta
    return float(refined_x), float(val)


def compute_rms(signal: np.ndarray) -> float:
    """Compute Root Mean Square (RMS) amplitude."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2)))


def detect_pitch_yin(
    signal: np.ndarray,
    sample_rate: int,
    min_freq: float = 40.0,
    max_freq: float = 1000.0,
    threshold: float = 0.20,
    min_rms: float = 0.0005
) -> Optional[float]:
    """
    Detect fundamental frequency (f0) using the YIN algorithm.
    Optimized difference function computation using FFT-based autocorrelation.
    Includes DC offset removal and sensitive energy gating.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)

    # Remove DC offset
    signal = signal - float(np.mean(signal))

    # Energy check
    rms = compute_rms(signal)
    if rms < min_rms:
        return None

    # Length check
    min_period = int(np.floor(sample_rate / max_freq))
    max_period = int(np.ceil(sample_rate / min_freq))
    if len(signal) < 2 * max_period:
        max_period = len(signal) // 2
        if max_period <= min_period:
            return None

    w_size = len(signal) - max_period
    if w_size <= 0:
        return None

    # Step 1: Difference function d(tau) using FFT autocorrelation:
    # d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    #        = sum x[j]^2 + sum x[j+tau]^2 - 2 * sum(x[j] * x[j+tau])
    x = signal
    n_fft = 1 << int(np.ceil(np.log2(w_size + max_period + w_size)))
    x_part = x[:w_size]
    X1 = np.fft.rfft(x_part, n=n_fft)
    X2 = np.fft.rfft(x[:w_size + max_period], n=n_fft)
    r = np.fft.irfft(X1.conj() * X2, n=n_fft)[:max_period + 1]

    # Cumulative energy terms
    x2 = x ** 2
    cum_x2 = np.cumsum(x2)
    term0 = np.sum(x_part ** 2)

    d = np.zeros(max_period + 1, dtype=np.float32)
    d[0] = 0.0
    for tau in range(1, max_period + 1):
        end_idx = w_size + tau - 1
        start_idx = tau - 1
        energy_shifted = cum_x2[end_idx] - cum_x2[start_idx]
        d[tau] = max(0.0, float(term0 + energy_shifted - 2.0 * r[tau]))

    # Step 2: Cumulative mean normalized difference function d'(tau)
    d_prime = np.zeros_like(d)
    d_prime[0] = 1.0
    running_sum = 0.0
    for tau in range(1, len(d)):
        running_sum += d[tau]
        if running_sum > 0:
            d_prime[tau] = d[tau] / (running_sum / tau)
        else:
            d_prime[tau] = 1.0

    # Step 3: Absolute threshold
    tau_chosen = -1
    for tau in range(min_period, max_period):
        if d_prime[tau] < threshold:
            # Look for local minimum
            while tau + 1 < max_period and d_prime[tau + 1] < d_prime[tau]:
                tau += 1
            tau_chosen = tau
            break

    # If no period met threshold, take global minimum if it has reasonable confidence
    if tau_chosen == -1:
        min_tau_idx = int(np.argmin(d_prime[min_period:max_period])) + min_period
        if d_prime[min_tau_idx] < 0.45:
            tau_chosen = min_tau_idx
        else:
            return None

    # Step 4: Parabolic interpolation
    refined_tau, _ = parabolic_interpolation(d_prime, tau_chosen)
    if refined_tau <= 0:
        return None

    freq = float(sample_rate / refined_tau)
    if min_freq <= freq <= max_freq:
        return freq
    return None


def compute_spectrum(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = 4096,
    min_freq: float = 40.0,
    max_freq: float = 700.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT magnitude spectrum (normalized 0..1) between min_freq and max_freq.
    Returns (freqs, magnitudes).
    """
    if len(signal) == 0:
        return np.array([]), np.array([])

    # DC removal
    sig_zero_mean = signal - float(np.mean(signal))

    if len(sig_zero_mean) < n_fft:
        pad_len = n_fft - len(sig_zero_mean)
        sig = np.pad(sig_zero_mean, (0, pad_len), mode="constant")
    else:
        sig = sig_zero_mean[-n_fft:]

    window = np.hanning(len(sig))
    windowed = sig * window
    fft_vals = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_band = freqs[mask]
    mag_band = fft_vals[mask]

    # Normalize magnitude to 0..1
    max_val = np.max(mag_band) if len(mag_band) > 0 else 1.0
    if max_val > 1e-6:
        mag_band = mag_band / max_val

    return freqs_band, mag_band


def analyze_audio_file(
    filepath: str,
    target_strings: List[Tuple[str, int, float]],
    frame_size: int = 4096,
    hop_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Optional[Tuple[int, str, float, float]]]:
    """
    Perform offline pitch analysis on an audio file.
    Returns:
      times: array of timestamp in seconds for each frame
      pitches: array of detected pitch (NaN if unvoiced/silent)
      dominant_pitch: overall estimated pitch (or None)
      closest_string_info: (index, label, target_hz, cents) for dominant pitch
    """
    data, sr = sf.read(filepath, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    total_samples = len(data)
    if total_samples < frame_size:
        frame_size = total_samples
        hop_size = max(1, frame_size // 2)

    times = []
    pitches = []
    energies = []

    for start in range(0, total_samples - frame_size + 1, hop_size):
        frame = data[start:start + frame_size]
        t = (start + frame_size / 2.0) / sr
        energy = compute_rms(frame)
        pitch = detect_pitch_yin(frame, sr, min_freq=40.0, max_freq=1000.0, min_rms=0.0005)

        times.append(t)
        energies.append(energy)
        pitches.append(pitch if pitch is not None else np.nan)

    times_arr = np.array(times, dtype=np.float32)
    pitches_arr = np.array(pitches, dtype=np.float32)
    energies_arr = np.array(energies, dtype=np.float32)

    # Determine dominant pitch:
    valid_mask = ~np.isnan(pitches_arr)
    if np.any(valid_mask):
        valid_pitches = pitches_arr[valid_mask]
        valid_energies = energies_arr[valid_mask]

        # Consider top 50% most energetic frames to avoid trailing decay / noise
        if len(valid_energies) > 4:
            e_thresh = np.percentile(valid_energies, 50)
            high_e_mask = valid_energies >= e_thresh
            if np.any(high_e_mask):
                dominant_pitch = float(np.median(valid_pitches[high_e_mask]))
            else:
                dominant_pitch = float(np.median(valid_pitches))
        else:
            dominant_pitch = float(np.median(valid_pitches))

        closest_info = find_closest_string(dominant_pitch, target_strings)
    else:
        dominant_pitch = None
        closest_info = None

    return times_arr, pitches_arr, dominant_pitch, closest_info
