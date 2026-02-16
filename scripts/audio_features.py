"""
Audio feature extraction helpers tailored for the babycry CRNN model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np

# Default preprocessing hyperparameters inferred from the original project.
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_FFT = 2048
DEFAULT_HOP = DEFAULT_FFT // 2
DEFAULT_MELS = 40
TARGET_FRAMES = 381


@dataclass(frozen=True)
class FeatureSpec:
    """
    Container for feature extraction hyperparameters.
    """

    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = DEFAULT_FFT
    hop_length: int = DEFAULT_HOP
    n_mels: int = DEFAULT_MELS
    target_frames: int = TARGET_FRAMES
    enforce_stereo: bool = True


def load_audio(path: Path, spec: FeatureSpec) -> Tuple[np.ndarray, int]:
    """
    Load audio from `path` and return an array shaped as (channels, samples).
    The loader keeps stereo information where available and optionally
    duplicates mono signals to two channels so the downstream model receives
    the expected input layout.
    """

    audio, sr = librosa.load(
        path,
        sr=spec.sample_rate,
        mono=False,
        dtype=np.float32,
    )

    if audio.ndim == 1:
        if spec.enforce_stereo:
            audio = np.stack([audio, audio], axis=0)
        else:
            audio = audio[np.newaxis, ...]

    return audio, sr


def _log_mel(channel_audio: np.ndarray, spec: FeatureSpec) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=channel_audio,
        sr=spec.sample_rate,
        n_fft=spec.n_fft,
        hop_length=spec.hop_length,
        n_mels=spec.n_mels,
        power=1.0,
    )
    return np.log(np.maximum(mel, 1e-10)).T


def extract_feature_matrix(path: Path, spec: FeatureSpec = FeatureSpec()) -> Tuple[np.ndarray, int]:
    """
    Produce a (frames, mel_bins * channels) matrix ready for scaling.
    """

    audio, _ = load_audio(path, spec)
    channel_features: Iterable[np.ndarray] = tuple(_log_mel(audio[ch], spec) for ch in range(audio.shape[0]))

    feature_matrix = np.concatenate(channel_features, axis=1)
    feature_matrix = _fit_to_target_frames(feature_matrix, spec.target_frames)

    return feature_matrix, audio.shape[0]


def _fit_to_target_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Pad or truncate along the time axis to guarantee a fixed-length tensor.
    """

    current_frames = features.shape[0]
    if current_frames == target_frames:
        return features
    if current_frames > target_frames:
        return features[:target_frames]

    pad = np.zeros((target_frames - current_frames, features.shape[1]), dtype=features.dtype)
    return np.vstack((features, pad))


def to_model_input(
    scaled_features: np.ndarray,
    num_channels: int,
    spec: FeatureSpec = FeatureSpec(),
) -> np.ndarray:
    """
    Convert a 2-D feature matrix into the (batch, channels, frames, mel_bins) tensor
    expected by the CRNN model.
    """

    frames, feature_dim = scaled_features.shape
    mel_bins = spec.n_mels
    expected_dim = num_channels * mel_bins

    if feature_dim != expected_dim:
        raise ValueError(
            f"Feature matrix has {feature_dim} columns but expected {expected_dim} "
            f"({num_channels} channels × {mel_bins} mel bins)."
        )

    reshaped = scaled_features.reshape(frames, num_channels, mel_bins)
    tensor = np.transpose(reshaped, (1, 0, 2))
    return tensor[np.newaxis, ...]
