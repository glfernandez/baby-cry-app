"""
Compute summary features that mirror donateacry-corpus_features_final.csv.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import librosa
import numpy as np
import pandas as pd


@dataclass
class FeatureSummary:
    amplitude_envelope_mean: float
    rms_mean: float
    zcr_mean: float
    stft_mean: float
    sc_mean: float
    sban_mean: float
    scon_mean: float
    mfccs13_mean: float
    del_mfccs13: float
    del2_mfccs13: float
    mel_spec: float
    mfccs20: float
    mfccs1: float
    mfccs2: float
    mfccs3: float
    mfccs4: float
    mfccs5: float
    mfccs6: float
    mfccs7: float
    mfccs8: float
    mfccs9: float
    mfccs10: float
    mfccs11: float
    mfccs12: float
    mfccs13: float


def _amplitude_envelope(y: np.ndarray, frame_length: int = 1024) -> np.ndarray:
    frame_count = int(np.ceil(len(y) / frame_length))
    envelope = np.zeros(frame_count, dtype=np.float32)
    for i in range(frame_count):
        start = i * frame_length
        end = min(start + frame_length, len(y))
        envelope[i] = np.max(np.abs(y[start:end]))
    return envelope


def _spectral_bandwidth(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]


def _spectral_contrast(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=0)


def summarize_wav(path: Path, sample_rate: int = 44100) -> Dict[str, float]:
    y, sr = librosa.load(path, sr=sample_rate, mono=True)
    amplitude = _amplitude_envelope(y)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    stft = librosa.stft(y)
    stft_magnitude = np.abs(stft)
    spectral_centroid = librosa.feature.spectral_centroid(S=stft_magnitude, sr=sr)[0]
    spectral_bandwidth = _spectral_bandwidth(y, sr)
    spectral_contrast = _spectral_contrast(y, sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec)
    mfcc = librosa.feature.mfcc(S=mel_spec_db, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc[:13])
    mfcc_delta2 = librosa.feature.delta(mfcc[:13], order=2)

    summary = FeatureSummary(
        amplitude_envelope_mean=float(np.mean(amplitude)),
        rms_mean=float(np.mean(rms)),
        zcr_mean=float(np.mean(zcr)),
        stft_mean=float(np.mean(stft_magnitude)),
        sc_mean=float(np.mean(spectral_centroid)),
        sban_mean=float(np.mean(spectral_bandwidth)),
        scon_mean=float(np.mean(spectral_contrast)),
        mfccs13_mean=float(np.mean(mfcc[:13])),
        del_mfccs13=float(np.mean(mfcc_delta)),
        del2_mfccs13=float(np.mean(mfcc_delta2)),
        mel_spec=float(np.mean(mel_spec_db)),
        mfccs20=float(np.mean(mfcc[19])),
        mfccs1=float(np.mean(mfcc[0])),
        mfccs2=float(np.mean(mfcc[1])),
        mfccs3=float(np.mean(mfcc[2])),
        mfccs4=float(np.mean(mfcc[3])),
        mfccs5=float(np.mean(mfcc[4])),
        mfccs6=float(np.mean(mfcc[5])),
        mfccs7=float(np.mean(mfcc[6])),
        mfccs8=float(np.mean(mfcc[7])),
        mfccs9=float(np.mean(mfcc[8])),
        mfccs10=float(np.mean(mfcc[9])),
        mfccs11=float(np.mean(mfcc[10])),
        mfccs12=float(np.mean(mfcc[11])),
        mfccs13=float(np.mean(mfcc[12])),
    )
    return asdict(summary)


def wavs_to_dataframe(paths: Sequence[Path], cry_reason: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for path in paths:
        features = summarize_wav(path)
        row = {
            "Cry_Audio_File": str(path.resolve()),
            "Cry_Reason": cry_reason,
        }
        row.update(features)
        rows.append(row)
    return pd.DataFrame(rows)


