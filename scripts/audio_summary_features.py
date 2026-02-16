"""
Feature-extraction helpers mirroring the DonateACry summary statistics.

This module matches the Android implementation found at
`android/app/src/main/java/com/aiyana/cry/ml/AudioFeatureExtraction.kt`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import librosa
import numpy as np


@dataclass(frozen=True)
class SummaryFeatureConfig:
    sample_rate: int = 44_100
    fft_size: int = 2048
    hop_length: int = 512
    envelope_frame: int = 1024
    mel_bins: int = 128
    mfcc_count: int = 20
    delta_window: int = 2
    eps: float = 1e-10


SUMMARY_COLUMNS: Sequence[str] = (
    "Amplitude_Envelope_Mean",
    "RMS_Mean",
    "ZCR_Mean",
    "STFT_Mean",
    "SC_Mean",
    "SBAN_Mean",
    "SCON_Mean",
    "MFCCs13Mean",
    "delMFCCs13",
    "del2MFCCs13",
    "MelSpec",
    "MFCCs20",
    "MFCCs1",
    "MFCCs2",
    "MFCCs3",
    "MFCCs4",
    "MFCCs5",
    "MFCCs6",
    "MFCCs7",
    "MFCCs8",
    "MFCCs9",
    "MFCCs10",
    "MFCCs11",
    "MFCCs12",
    "MFCCs13",
)


def load_mono_audio(path: Path, *, config: SummaryFeatureConfig) -> np.ndarray:
    signal, _ = librosa.load(path, sr=config.sample_rate, mono=True, dtype=np.float32)
    if signal.ndim != 1:
        raise ValueError(f"Expected mono signal, got shape {signal.shape}")
    return signal


def amplitude_envelope_mean(samples: np.ndarray, *, config: SummaryFeatureConfig) -> float:
    if samples.size == 0:
        return 0.0
    frame = config.envelope_frame
    maxima = [
        float(np.max(np.abs(samples[i : i + frame])))
        for i in range(0, samples.size, frame)
    ]
    return float(np.mean(maxima)) if maxima else 0.0


def frame_count(length: int, *, config: SummaryFeatureConfig) -> int:
    if length <= 0:
        return 1
    return max(1, 1 + math.ceil((length - config.fft_size) / config.hop_length))


def iter_frames(samples: np.ndarray, *, config: SummaryFeatureConfig) -> Iterator[np.ndarray]:
    total_frames = frame_count(samples.size, config=config)
    pad_length = max(0, (total_frames - 1) * config.hop_length + config.fft_size - samples.size)
    padded = np.pad(samples, (0, pad_length), mode="constant")
    for start in range(0, total_frames * config.hop_length, config.hop_length):
        end = start + config.fft_size
        yield padded[start:end]


def rms_per_frame(samples: np.ndarray, *, config: SummaryFeatureConfig) -> np.ndarray:
    rms_values = [
        float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))) for frame in iter_frames(samples, config=config)
    ]
    return np.asarray(rms_values, dtype=np.float32)


def zero_crossing_rate_per_frame(samples: np.ndarray, *, config: SummaryFeatureConfig) -> np.ndarray:
    zcr: list[float] = []
    for frame in iter_frames(samples, config=config):
        if frame.size == 0:
            zcr.append(0.0)
            continue
        crossings = 0
        prev = float(frame[0])
        for current in frame[1:]:
            current = float(current)
            if prev == 0.0:
                prev = current
                continue
            if current == 0.0:
                continue
            if (prev > 0.0 and current < 0.0) or (prev < 0.0 and current > 0.0):
                crossings += 1
            prev = current
        zcr.append(crossings / frame.size)
    return np.asarray(zcr, dtype=np.float32)


def hann_window(config: SummaryFeatureConfig) -> np.ndarray:
    return np.hanning(config.fft_size)


def compute_stft(samples: np.ndarray, *, config: SummaryFeatureConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = hann_window(config)
    n_bins = config.fft_size // 2 + 1
    frames = list(iter_frames(samples, config=config))
    magnitudes = np.empty((len(frames), n_bins), dtype=np.float32)
    power = np.empty_like(magnitudes)

    for idx, frame in enumerate(frames):
        windowed = frame * window
        spectrum = np.fft.rfft(windowed.astype(np.float64), n=config.fft_size)
        mag = np.abs(spectrum).astype(np.float32)
        magnitudes[idx, :] = mag
        power[idx, :] = (mag ** 2).astype(np.float32)

    freqs = np.linspace(0.0, config.sample_rate / 2.0, n_bins, dtype=np.float64)
    return magnitudes, power, freqs


def mean_of_matrix(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.mean(matrix))


def spectral_centroid_per_frame(magnitudes: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    numerator = np.sum(magnitudes * freqs, axis=1)
    denominator = np.sum(magnitudes, axis=1)
    centroid = np.zeros_like(numerator)
    valid = denominator > 0
    centroid[valid] = numerator[valid] / denominator[valid]
    return centroid


def spectral_centroid_mean(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    if magnitudes.size == 0:
        return 0.0
    denominator = np.sum(magnitudes, axis=1)
    valid = denominator > 0
    if not np.any(valid):
        return 0.0
    centroid = spectral_centroid_per_frame(magnitudes, freqs)
    return float(np.mean(centroid[valid]))


def spectral_bandwidth_mean(magnitudes: np.ndarray, freqs: np.ndarray) -> float:
    if magnitudes.size == 0:
        return 0.0
    centroid = spectral_centroid_per_frame(magnitudes, freqs)
    variances = np.sum(magnitudes * (freqs - centroid[:, None]) ** 2, axis=1)
    denominator = np.sum(magnitudes, axis=1)
    valid = denominator > 0
    if not np.any(valid):
        return 0.0
    bandwidth = np.zeros_like(variances)
    bandwidth[valid] = np.sqrt(variances[valid] / denominator[valid])
    return float(np.mean(bandwidth[valid]))


def spectral_contrast_mean(
    magnitudes: np.ndarray, freqs: np.ndarray, *, sample_rate: int
) -> float:
    if magnitudes.size == 0:
        return 0.0

    n_bands = 6
    f_min = 200.0
    upper = sample_rate / 2.0

    band_edges = np.empty(n_bands + 2, dtype=np.float64)
    band_edges[0] = max(freqs[1], f_min / 2.0)
    for idx in range(1, n_bands + 1):
        band_edges[idx] = min(f_min * (2.0 ** (idx - 1)), upper)
    band_edges[-1] = upper

    contrasts: list[float] = []
    for frame in magnitudes:
        band_values: list[float] = []
        for band in range(n_bands):
            low, high = band_edges[band], band_edges[band + 1]
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                continue
            selected = frame[mask].astype(np.float64)
            if selected.size == 0:
                continue
            max_val = float(np.max(selected))
            positive = selected[selected > 0.0]
            min_val = float(np.min(positive)) if positive.size else max_val
            if max_val <= 0.0:
                continue
            contrast = 10.0 * math.log10((max_val + 1e-10) / (min_val + 1e-10))
            band_values.append(contrast)
        contrasts.append(float(np.mean(band_values) if band_values else 0.0))
    return float(np.mean(contrasts))


def mel_filter_bank(config: SummaryFeatureConfig) -> np.ndarray:
    n_bins = config.fft_size // 2 + 1
    f_min = 0.0
    f_max = config.sample_rate / 2.0
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, config.mel_bins + 2)
    bin_freqs = np.linspace(f_min, f_max, n_bins)
    filters = np.zeros((config.mel_bins, n_bins), dtype=np.float64)
    for m in range(config.mel_bins):
        left = mel_points[m]
        center = mel_points[m + 1]
        right = mel_points[m + 2]
        left_hz, center_hz, right_hz = mel_to_hz(left), mel_to_hz(center), mel_to_hz(right)
        for i, freq in enumerate(bin_freqs):
            if freq < left_hz:
                weight = 0.0
            elif freq <= center_hz:
                denom = max(center_hz - left_hz, 1e-6)
                weight = (freq - left_hz) / denom
            elif freq <= right_hz:
                denom = max(right_hz - center_hz, 1e-6)
                weight = (right_hz - freq) / denom
            else:
                weight = 0.0
            filters[m, i] = weight
    return filters


@lru_cache(maxsize=1)
def cached_mel_filter_bank(config: SummaryFeatureConfig) -> np.ndarray:
    return mel_filter_bank(config)


def mel_spectrogram_db(power: np.ndarray, *, config: SummaryFeatureConfig) -> np.ndarray:
    filters = cached_mel_filter_bank(config)
    mel_spec = power @ filters.T
    return power_to_db(mel_spec, eps=config.eps)


def power_to_db(x: np.ndarray, *, eps: float) -> np.ndarray:
    ref = 1.0
    return 10.0 * np.log10(np.maximum(eps, x / ref))


@lru_cache(maxsize=1)
def cached_dct_matrix(config: SummaryFeatureConfig) -> np.ndarray:
    m = config.mfcc_count
    n = config.mel_bins
    matrix = np.zeros((m, n), dtype=np.float64)
    factor = math.sqrt(2.0 / n)
    for i in range(m):
        for j in range(n):
            matrix[i, j] = factor * math.cos(math.pi * i * (j + 0.5) / n)
    matrix[0, :] *= 1.0 / math.sqrt(2.0)
    return matrix


def compute_mfcc(mel_db: np.ndarray, *, config: SummaryFeatureConfig) -> np.ndarray:
    dct = cached_dct_matrix(config)
    return mel_db @ dct.T


def summarize_mfcc(mfcc: np.ndarray, *, config: SummaryFeatureConfig) -> tuple[np.ndarray, float]:
    if mfcc.size == 0:
        return np.zeros(config.mfcc_count, dtype=np.float32), 0.0
    coeff_means = mfcc.mean(axis=0)
    first13 = mfcc[:, :13]
    first13_mean = float(first13.mean()) if first13.size else 0.0
    return coeff_means.astype(np.float32), first13_mean


def compute_delta(data: np.ndarray, *, config: SummaryFeatureConfig) -> np.ndarray:
    if data.size == 0:
        return np.zeros_like(data)
    window = config.delta_window
    frames, coeffs = data.shape
    denom = 2 * sum(n * n for n in range(1, window + 1))
    padded = np.pad(data, ((window, window), (0, 0)), mode="edge")
    deltas = np.zeros_like(data, dtype=np.float32)
    for t in range(frames):
        acc = np.zeros(coeffs, dtype=np.float64)
        for n in range(1, window + 1):
            acc += n * (padded[t + window + n] - padded[t + window - n])
        deltas[t] = (acc / denom).astype(np.float32)
    return deltas


def feature_vector_from_audio(samples: np.ndarray, sample_rate: int, *, config: SummaryFeatureConfig) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(len(SUMMARY_COLUMNS), dtype=np.float32)

    amplitude_mean = amplitude_envelope_mean(samples, config=config)
    rms_values = rms_per_frame(samples, config=config)
    zcr_values = zero_crossing_rate_per_frame(samples, config=config)

    magnitudes, power, freqs = compute_stft(samples, config=config)
    stft_mean = mean_of_matrix(magnitudes)
    spectral_centroid = spectral_centroid_mean(magnitudes, freqs)
    spectral_bandwidth = spectral_bandwidth_mean(magnitudes, freqs)
    spectral_contrast = spectral_contrast_mean(magnitudes, freqs, sample_rate=sample_rate)

    mel_db = mel_spectrogram_db(power, config=config)
    mel_spec_mean = mean_of_matrix(mel_db)

    mfcc_frames = compute_mfcc(mel_db, config=config)
    coeff_means, first13_mean = summarize_mfcc(mfcc_frames, config=config)

    delta1 = compute_delta(mfcc_frames[:, :13], config=config)
    delta2 = compute_delta(delta1, config=config)

    delta_summary = float(delta1.mean()) if delta1.size else 0.0
    delta2_summary = float(delta2.mean()) if delta2.size else 0.0

    features = np.zeros(len(SUMMARY_COLUMNS), dtype=np.float32)
    idx = 0
    features[idx] = amplitude_mean
    idx += 1
    features[idx] = float(rms_values.mean()) if rms_values.size else 0.0
    idx += 1
    features[idx] = float(zcr_values.mean()) if zcr_values.size else 0.0
    idx += 1
    features[idx] = stft_mean
    idx += 1
    features[idx] = spectral_centroid
    idx += 1
    features[idx] = spectral_bandwidth
    idx += 1
    features[idx] = spectral_contrast
    idx += 1
    features[idx] = first13_mean
    idx += 1
    features[idx] = delta_summary
    idx += 1
    features[idx] = delta2_summary
    idx += 1
    features[idx] = mel_spec_mean
    idx += 1
    features[idx] = coeff_means[19] if coeff_means.size >= 20 else 0.0
    idx += 1
    for coeff in coeff_means[:13]:
        features[idx] = coeff
        idx += 1
    return features


def extract_summary_features(path: Path, *, config: SummaryFeatureConfig = SummaryFeatureConfig()) -> np.ndarray:
    samples = load_mono_audio(path, config=config)
    return feature_vector_from_audio(samples, config.sample_rate, config=config)


def hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def wavs_to_rows(paths: Iterable[Path], *, config: SummaryFeatureConfig) -> Iterator[tuple[Path, np.ndarray]]:
    for path in paths:
        yield path, extract_summary_features(path, config=config)
