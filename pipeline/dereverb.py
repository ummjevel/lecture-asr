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

# 청크 분할 파라미터 (메모리 제한)
_CHUNK_SEC = 300  # 5분 단위 청크
_OVERLAP_SEC = 2  # 2초 오버랩 (크로스페이드)


def _process_chunk(
    chunk: np.ndarray, params: dict,
) -> np.ndarray:
    """단일 청크에 WPE 역반향 제거를 적용한다."""
    chunk_len = len(chunk)
    chunk_64 = chunk.astype(np.float64)
    signal = chunk_64[np.newaxis, :]

    Y = stft(signal, size=_STFT_SIZE, shift=_STFT_SHIFT)
    Z = wpe(
        Y,
        taps=params["taps"],
        delay=params["delay"],
        iterations=params["iterations"],
    )
    result = istft(Z, size=_STFT_SIZE, shift=_STFT_SHIFT)
    return result[0, :chunk_len].astype(np.float32)


def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    **kwargs,
) -> np.ndarray:
    """역반향 제거를 적용한다.

    5분 단위 청크로 분할 처리하여 메모리 사용량을 제한한다.

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
    original_len = len(audio)

    chunk_samples = _CHUNK_SEC * sr
    overlap_samples = _OVERLAP_SEC * sr

    # 짧은 오디오는 한번에 처리
    if original_len <= chunk_samples + overlap_samples:
        logger.info("dereverb: preset=%s (단일 청크)", preset)
        return _process_chunk(audio, params)

    # 청크 분할 + 크로스페이드
    n_chunks = 0
    pos = 0
    while pos < original_len:
        n_chunks += 1
        pos += chunk_samples
    logger.info("dereverb: preset=%s, %d 청크 (%.0f초 단위)", preset, n_chunks, _CHUNK_SEC)

    output = np.zeros(original_len, dtype=np.float32)
    pos = 0

    while pos < original_len:
        # 청크 범위 (오버랩 포함)
        end = min(pos + chunk_samples + overlap_samples, original_len)
        chunk = audio[pos:end]

        processed = _process_chunk(chunk, params)

        if pos == 0:
            # 첫 청크: 그대로 복사
            output[:len(processed)] = processed
        else:
            # 오버랩 구간 크로스페이드
            fade_len = min(overlap_samples, len(processed), original_len - pos)
            if fade_len > 0:
                fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                output[pos:pos + fade_len] = (
                    output[pos:pos + fade_len] * fade_out
                    + processed[:fade_len] * fade_in
                )
            # 오버랩 이후 구간
            remaining = processed[fade_len:]
            output[pos + fade_len:pos + fade_len + len(remaining)] = remaining

        pos += chunk_samples

    logger.info("dereverb done: %d samples", original_len)
    return output
