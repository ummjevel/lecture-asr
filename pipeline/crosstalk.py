"""에너지 기반 크로스토크(배경 대화) 감쇠.

주 화자(교수)는 일정하고 큰 에너지, 배경 대화는 작고 산발적.
프레임별 에너지를 측정하여 주 화자 대비 낮은 에너지 구간을 감쇠한다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 프리셋별 파라미터
_PRESETS = {
    "light": {"attenuation": 0.3, "threshold_ratio": 0.3, "frame_ms": 30},
    "normal": {"attenuation": 0.15, "threshold_ratio": 0.4, "frame_ms": 30},
    "strong": {"attenuation": 0.05, "threshold_ratio": 0.5, "frame_ms": 30},
}


def _compute_frame_energies(
    audio: np.ndarray, frame_len: int,
) -> np.ndarray:
    """프레임별 RMS 에너지를 계산한다."""
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return np.array([])
    # 프레임 단위로 reshape 가능한 만큼만 사용
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    **kwargs,
) -> np.ndarray:
    """에너지 기반 크로스토크 감쇠.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트.
        preset: "light" | "normal" | "strong".

    Returns:
        크로스토크 감쇠된 오디오 ndarray.
    """
    if preset == "light":
        # light에서는 스킵 (마이크 사용 → 크로스토크 적음)
        return audio

    params = _PRESETS.get(preset, _PRESETS["normal"])
    frame_ms = params["frame_ms"]
    threshold_ratio = params["threshold_ratio"]
    attenuation = params["attenuation"]

    frame_len = int(sr * frame_ms / 1000)
    energies = _compute_frame_energies(audio, frame_len)

    if len(energies) == 0:
        return audio

    # 주 화자 에너지 추정: 상위 50% 에너지의 중앙값
    sorted_e = np.sort(energies)
    upper_half = sorted_e[len(sorted_e) // 2:]
    if len(upper_half) == 0:
        return audio
    primary_energy = np.median(upper_half)

    if primary_energy < 1e-8:
        return audio

    threshold = primary_energy * threshold_ratio

    # 감쇠 적용
    result = audio.copy()
    n_frames = len(energies)
    suppressed = 0

    for i in range(n_frames):
        if energies[i] < threshold:
            start = i * frame_len
            end = min((i + 1) * frame_len, len(result))

            # 부드러운 감쇠 (hard cut 대신 fade)
            fade_len = min(frame_len // 4, end - start)
            gain = np.ones(end - start, dtype=np.float32) * attenuation

            # fade-in/out로 자연스럽게
            if fade_len > 0:
                gain[:fade_len] = np.linspace(1.0, attenuation, fade_len)
                gain[-fade_len:] = np.linspace(attenuation, 1.0, fade_len)

            result[start:end] *= gain
            suppressed += 1

    logger.info(
        "crosstalk: %d/%d 프레임 감쇠 (threshold=%.4f, preset=%s)",
        suppressed, n_frames, threshold, preset,
    )
    return result
