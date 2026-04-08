"""AGC (Automatic Gain Control) — pyagc 동적 볼륨 정규화."""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# pyagc 선택적 의존성
_HAS_PYAGC = False
try:
    import pyagc  # noqa: F401

    _HAS_PYAGC = True
except ImportError:
    pass

# 프리셋별 AGC 파라미터
_AGC_PARAMS: dict[str, dict] = {
    "normal": {
        "target_level_db": -20.0,
        "max_gain_db": 20.0,
        "attack_time": 0.01,
        "release_time": 0.3,
    },
    "strong": {
        "target_level_db": -18.0,
        "max_gain_db": 30.0,
        "attack_time": 0.005,
        "release_time": 0.2,
    },
}


def _rms_normalize(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """간단한 RMS 기반 정규화 (pyagc 폴백)."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio

    target_rms = 10.0 ** (target_db / 20.0)
    gain = target_rms / rms

    # 최대 게인 제한 (30dB)
    max_gain = 10.0 ** (30.0 / 20.0)
    gain = min(gain, max_gain)

    result = audio * gain

    # 클리핑 방지
    peak = np.max(np.abs(result))
    if peak > 0.99:
        result = result * (0.99 / peak)

    return result.astype(np.float32)


def _agc_pyagc(
    audio: np.ndarray, sr: int, preset: str
) -> np.ndarray:
    """pyagc 라이브러리로 동적 볼륨 정규화."""
    params = _AGC_PARAMS.get(preset, _AGC_PARAMS["normal"])

    agc = pyagc.AGC(
        sample_rate=sr,
        target_level_db=params["target_level_db"],
        max_gain_db=params["max_gain_db"],
        attack_time=params["attack_time"],
        release_time=params["release_time"],
    )
    result = agc.process(audio)
    return np.asarray(result, dtype=np.float32)


def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    **kwargs,
) -> np.ndarray:
    """동적 볼륨 정규화(AGC)를 적용한다.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트 (16000).
        preset: "light" (스킵) | "normal" | "strong".

    Returns:
        볼륨 정규화된 오디오 (float32).
    """
    if preset == "light":
        logger.info("agc: light 프리셋 — 스킵")
        return audio

    params = _AGC_PARAMS.get(preset, _AGC_PARAMS["normal"])

    if _HAS_PYAGC:
        try:
            audio = _agc_pyagc(audio, sr, preset)
            logger.info("agc: pyagc 적용 (preset=%s)", preset)
            return audio
        except Exception as exc:
            logger.warning("pyagc 실패, RMS 정규화로 폴백: %s", exc)

    # 폴백: RMS 기반 정규화
    if not _HAS_PYAGC:
        warnings.warn(
            "pyagc 미설치. RMS 기반 정규화로 대체합니다. "
            "설치: pip install pyagc",
            stacklevel=2,
        )

    audio = _rms_normalize(audio, target_db=params["target_level_db"])
    logger.info("agc: RMS 정규화 폴백 (preset=%s)", preset)
    return audio
