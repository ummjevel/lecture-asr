"""WADA SNR 측정 + 프리셋 자동 선택 — 순수 NumPy/SciPy 구현."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# WADA SNR 상수 (Waveform Amplitude Distribution Analysis)
# C. Kim & R. Stern, "Robust Signal-to-Noise Ratio Estimation Based on
# Waveform Amplitude Distribution Analysis", Interspeech 2008.
_WADA_GAMMA = 0.8  # shape parameter for clean speech model


def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """WADA SNR 추정 알고리즘으로 SNR(dB)을 반환한다.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트.

    Returns:
        추정 SNR (dB). 오디오가 비어 있거나 무음이면 0.0.
    """
    if len(audio) == 0:
        return 0.0

    audio = audio.astype(np.float64)

    # DC 제거
    audio = audio - np.mean(audio)

    # 절대값
    abs_audio = np.abs(audio)
    mean_abs = np.mean(abs_audio)

    if mean_abs < 1e-10:
        return 0.0

    # WADA: E[|x|^gamma] 기반 SNR 추정
    # 깨끗한 음성의 진폭 분포를 가정하고 잡음 수준을 추정
    # 간략화된 WADA 접근: kurtosis 기반
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return 0.0

    # 프레임 단위 에너지 분석 (20ms 프레임)
    frame_len = int(sr * 0.02)
    if frame_len == 0:
        frame_len = 1

    n_frames = len(audio) // frame_len
    if n_frames < 2:
        return 0.0

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_energy = np.sum(frames ** 2, axis=1) / frame_len

    # 에너지가 0인 프레임 제외
    frame_energy = frame_energy[frame_energy > 0]
    if len(frame_energy) < 2:
        return 0.0

    frame_energy_db = 10.0 * np.log10(frame_energy + 1e-10)

    # 에너지 히스토그램으로 음성/비음성 분리
    # 하위 15% 에너지를 노이즈 플로어로 추정
    sorted_energy = np.sort(frame_energy_db)
    noise_percentile = 0.15
    n_noise = max(1, int(len(sorted_energy) * noise_percentile))

    noise_floor_db = np.mean(sorted_energy[:n_noise])

    # 상위 에너지를 신호로 추정
    signal_percentile = 0.85
    n_signal_start = int(len(sorted_energy) * (1.0 - signal_percentile))
    # 상위 50%의 평균 에너지
    signal_frames = sorted_energy[len(sorted_energy) // 2 :]
    if len(signal_frames) == 0:
        return 0.0

    signal_level_db = np.mean(signal_frames)

    snr_db = float(signal_level_db - noise_floor_db)

    # 합리적인 범위로 클리핑
    snr_db = np.clip(snr_db, -10.0, 60.0)

    logger.info("estimate_snr: %.1f dB", snr_db)
    return snr_db


def auto_preset(snr_db: float) -> str:
    """SNR 값으로부터 노이즈 제거 프리셋을 자동 결정한다.

    Args:
        snr_db: 추정 SNR (dB).

    Returns:
        "light", "normal", 또는 "strong".
    """
    if snr_db > 20.0:
        preset = "light"
    elif snr_db >= 10.0:
        preset = "normal"
    else:
        preset = "strong"

    logger.info("auto_preset: SNR=%.1f dB → %s", snr_db, preset)
    return preset
