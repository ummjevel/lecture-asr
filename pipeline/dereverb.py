"""nara_wpe 역반향 제거 — 강의실 반향(reverberation) 제거."""

from __future__ import annotations

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)

# nara_wpe 선택적 의존성
try:
    from nara_wpe.wpe import wpe_v6 as wpe
    from nara_wpe.utils import stft, istft

    _HAS_NARA_WPE = True
except ImportError:
    _HAS_NARA_WPE = False

# 프리셋별 WPE 파라미터
_WPE_PARAMS: dict[str, dict] = {
    "normal": {
        "taps": 10,
        "delay": 3,
        "iterations": 3,
    },
    "strong": {
        "taps": 20,
        "delay": 2,
        "iterations": 5,
    },
}

# STFT 파라미터
_STFT_SIZE = 512
_STFT_SHIFT = 128


def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    **kwargs,
) -> np.ndarray:
    """역반향 제거를 적용한다.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트 (16000).
        preset: "light" (스킵) | "normal" | "strong".

    Returns:
        역반향 제거된 오디오 (float32, 동일 길이).
    """
    if preset == "light":
        logger.info("dereverb: light 프리셋 — 스킵")
        return audio

    if not _HAS_NARA_WPE:
        warnings.warn(
            "nara_wpe가 설치되지 않아 역반향 제거를 건너뜁니다. "
            "설치: pip install nara_wpe",
            stacklevel=2,
        )
        logger.warning("dereverb: nara_wpe 미설치 — 스킵")
        return audio

    params = _WPE_PARAMS.get(preset, _WPE_PARAMS["normal"])
    logger.info("dereverb: preset=%s, params=%s", preset, params)

    original_len = len(audio)
    audio_64 = audio.astype(np.float64)

    # mono → (1, T) 형태로 변환 (nara_wpe는 다채널 입력을 기대)
    signal = audio_64[np.newaxis, :]

    # STFT
    Y = stft(signal, size=_STFT_SIZE, shift=_STFT_SHIFT)
    # Y shape: (channels, frames, freq_bins)

    # WPE 역반향 제거
    Z = wpe(
        Y,
        taps=params["taps"],
        delay=params["delay"],
        iterations=params["iterations"],
    )

    # iSTFT
    result = istft(Z, size=_STFT_SIZE, shift=_STFT_SHIFT)

    # mono 채널 추출 + 원래 길이로 트리밍
    out = result[0, :original_len].astype(np.float32)

    logger.info("dereverb done: %d samples", len(out))
    return out
