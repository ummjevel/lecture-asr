"""전처리 파이프라인 — run_preprocess() + save_preprocessed() 헬퍼."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from pipeline import converter, snr, dereverb, denoiser, click_remover, crosstalk, agc

logger = logging.getLogger(__name__)

# on_progress 콜백 타입
ProgressCallback = Callable[[str, float, str], None]


def _noop_progress(step: str, percent: float, message: str) -> None:
    """기본 no-op 콜백."""
    pass


def run_preprocess(
    input_path: str,
    preset: str = "normal",
    denoise_engine: str = "auto",
    on_progress: ProgressCallback | None = None,
) -> tuple[np.ndarray, int]:
    """전처리 파이프라인 전체를 순차 실행한다.

    convert → snr 측정 → dereverb → denoise → agc

    Args:
        input_path: 입력 오디오 파일 경로 (m4a/mp3/wav 등).
        preset: "light" | "normal" | "strong" | "auto".
        denoise_engine: "auto" (DeepFilterNet→폴백) | "lightweight" (noisereduce만).
        on_progress: 진행 콜백 ``(step, percent, message) -> None``.

    Returns:
        (audio, sr) — 전처리 완료된 float32 ndarray와 샘플레이트(16000).
    """
    cb = on_progress or _noop_progress

    # 1. 오디오 변환
    cb("convert", 5.0, "ffmpeg 변환 중... (시간이 걸릴 수 있습니다)")
    audio, sr = converter.convert(input_path)
    cb("convert", 100.0, f"변환 완료: {len(audio) / sr:.1f}초, {sr}Hz")

    # 2. SNR 측정
    cb("snr", 10.0, "SNR 측정 중")
    snr_db = snr.estimate_snr(audio, sr)
    effective_preset = preset

    if preset == "auto":
        effective_preset = snr.auto_preset(snr_db)
        cb("snr", 100.0, f"SNR={snr_db:.1f}dB → 프리셋: {effective_preset}")
    else:
        cb("snr", 100.0, f"SNR={snr_db:.1f}dB (프리셋: {preset} 수동)")

    logger.info("preset=%s, effective=%s, snr=%.1fdB", preset, effective_preset, snr_db)

    # 3. 역반향 제거
    cb("dereverb", 5.0, "WPE 역반향 제거 중... (시간이 걸릴 수 있습니다)")
    audio = dereverb.process(audio, sr, preset=effective_preset)
    cb("dereverb", 100.0, "역반향 제거 완료")

    # 4. 노이즈 제거
    engine_label = "noisereduce (경량)" if denoise_engine == "lightweight" else "자동"
    cb("denoise", 5.0, f"노이즈 제거 중... ({engine_label})")
    audio = denoiser.process(audio, sr, preset=effective_preset, engine=denoise_engine)
    cb("denoise", 100.0, "노이즈 제거 완료")

    # 4.5 클릭 제거 (키보드/마우스 충격 소음)
    if effective_preset != "light":
        cb("denoise", 100.0, "클릭 소음 제거 중...")
        audio = click_remover.process(audio, sr, preset=effective_preset)

    # 4.6 크로스토크 감쇠 (배경 대화)
    if effective_preset != "light":
        cb("denoise", 100.0, "크로스토크 감쇠 중...")
        audio = crosstalk.process(audio, sr, preset=effective_preset)

    # 5. AGC
    cb("agc", 5.0, "볼륨 정규화 중...")
    audio = agc.process(audio, sr, preset=effective_preset)
    cb("agc", 100.0, "볼륨 정규화 완료")

    # 전처리 완료 — 무거운 모델 메모리 해제 (ASR 모델 로딩 전)
    denoiser.release_model()
    import gc, sys
    gc.collect()
    # torch가 이미 로드된 경우에만 MPS 캐시 해제 (torch를 새로 import하지 않음)
    if "torch" in sys.modules:
        try:
            torch = sys.modules["torch"]
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
    logger.info("전처리 모델 메모리 해제 완료")

    return audio, sr


def save_preprocessed(
    audio: np.ndarray,
    sr: int,
    output_path: str | None = None,
) -> str:
    """전처리된 ndarray를 WAV 파일로 저장한다 (ASR 브릿지).

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트.
        output_path: 저장 경로. None이면 임시 파일 생성.

    Returns:
        저장된 WAV 파일의 절대 경로.
    """
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    output_path = str(Path(output_path).resolve())
    sf.write(output_path, audio, sr, subtype="PCM_16")
    logger.info("save_preprocessed: %s (%.1fs)", output_path, len(audio) / sr)
    return output_path
