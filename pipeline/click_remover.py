"""키보드/마우스 클릭 등 충격(impulsive) 소음 제거.

onset 기반 transient 감지 → 해당 구간 보간 처리.
librosa 미설치 시 scipy 기반 폴백.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 프리셋별 파라미터
_PRESETS = {
    "light": {"threshold": 4.0, "margin_ms": 5},
    "normal": {"threshold": 3.0, "margin_ms": 8},
    "strong": {"threshold": 2.0, "margin_ms": 12},
}


def _detect_clicks_librosa(
    audio: np.ndarray, sr: int, threshold: float, margin_ms: int,
) -> list[tuple[int, int]]:
    """librosa onset 기반 클릭 구간 감지."""
    import librosa

    hop_length = 512
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    if len(onset_env) == 0:
        return []

    median_val = np.median(onset_env)
    if median_val < 1e-8:
        return []

    click_frames = np.where(onset_env > threshold * median_val)[0]

    margin_samples = int(sr * margin_ms / 1000)
    regions: list[tuple[int, int]] = []
    for frame in click_frames:
        center = frame * hop_length
        start = max(0, center - margin_samples)
        end = min(len(audio), center + margin_samples)
        regions.append((start, end))

    # 겹치는 구간 병합
    if not regions:
        return []
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _detect_clicks_scipy(
    audio: np.ndarray, sr: int, threshold: float, margin_ms: int,
) -> list[tuple[int, int]]:
    """scipy 기반 폴백 — 에너지 변화율로 클릭 감지."""
    from scipy.signal import medfilt

    frame_len = int(sr * 0.01)  # 10ms 프레임
    n_frames = len(audio) // frame_len

    if n_frames < 3:
        return []

    energies = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    # 에너지 변화율 (delta)
    delta = np.abs(np.diff(energies))
    median_delta = np.median(delta)
    if median_delta < 1e-8:
        return []

    click_frames = np.where(delta > threshold * median_delta)[0]

    margin_samples = int(sr * margin_ms / 1000)
    regions: list[tuple[int, int]] = []
    for frame in click_frames:
        center = frame * frame_len
        start = max(0, center - margin_samples)
        end = min(len(audio), center + margin_samples)
        regions.append((start, end))

    if not regions:
        return []
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    return merged


def _interpolate_regions(
    audio: np.ndarray, regions: list[tuple[int, int]],
) -> np.ndarray:
    """감지된 클릭 구간을 선형 보간으로 대체."""
    result = audio.copy()
    for start, end in regions:
        if end - start < 2:
            continue
        left_val = result[start]
        right_val = result[min(end, len(result) - 1)]
        result[start:end] = np.linspace(left_val, right_val, end - start)
    return result


def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    **kwargs,
) -> np.ndarray:
    """충격 소음(키보드/마우스 클릭) 제거.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트.
        preset: "light" | "normal" | "strong".

    Returns:
        클릭 제거된 오디오 ndarray.
    """
    if preset == "light":
        # light에서는 스킵 (마이크 사용 강의 → 클릭 거의 없음)
        return audio

    params = _PRESETS.get(preset, _PRESETS["normal"])
    threshold = params["threshold"]
    margin_ms = params["margin_ms"]

    # librosa 우선, 없으면 scipy 폴백
    try:
        regions = _detect_clicks_librosa(audio, sr, threshold, margin_ms)
        backend = "librosa"
    except ImportError:
        regions = _detect_clicks_scipy(audio, sr, threshold, margin_ms)
        backend = "scipy"

    if not regions:
        logger.info("click_remover: 클릭 미감지 (backend=%s)", backend)
        return audio

    result = _interpolate_regions(audio, regions)
    logger.info(
        "click_remover: %d개 클릭 구간 제거 (backend=%s, preset=%s)",
        len(regions), backend, preset,
    )
    return result
