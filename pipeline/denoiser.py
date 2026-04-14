"""노이즈 제거 — DeepFilterNet3 (기본) / noisereduce (폴백)."""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# 선택적 의존성 탐색 (lazy — import 시점에 torch 로드 방지)
# ------------------------------------------------------------------
_HAS_DEEPFILTER: bool | None = None  # None = 미확인
_HAS_NOISEREDUCE: bool | None = None


def _check_deepfilter() -> bool:
    global _HAS_DEEPFILTER
    if _HAS_DEEPFILTER is None:
        try:
            from df.enhance import enhance, init_df  # noqa: F401
            _HAS_DEEPFILTER = True
        except ImportError:
            _HAS_DEEPFILTER = False
    return _HAS_DEEPFILTER


def _check_noisereduce() -> bool:
    global _HAS_NOISEREDUCE
    if _HAS_NOISEREDUCE is None:
        try:
            import noisereduce  # noqa: F401
            _HAS_NOISEREDUCE = True
        except ImportError:
            _HAS_NOISEREDUCE = False
    return _HAS_NOISEREDUCE

# ------------------------------------------------------------------
# 프리셋별 파라미터
# ------------------------------------------------------------------
_DEEPFILTER_PARAMS: dict[str, dict] = {
    "light": {"atten_lim_db": 12},
    "normal": {"atten_lim_db": 20},
    "strong": {"atten_lim_db": 30},
}

_NOISEREDUCE_PARAMS: dict[str, dict] = {
    "light": {
        "stationary": True,
        "prop_decrease": 0.5,
    },
    "normal": {
        "stationary": False,
        "prop_decrease": 0.75,
    },
    "strong": {
        "stationary": False,
        "prop_decrease": 1.0,
    },
}


# ------------------------------------------------------------------
# 하이패스 필터 (strong 프리셋: 키보드 저주파 제거)
# ------------------------------------------------------------------
def _highpass(audio: np.ndarray, sr: int, cutoff: float = 80.0) -> np.ndarray:
    """Butterworth 하이패스 필터 (80Hz)."""
    sos = butter(5, cutoff, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ------------------------------------------------------------------
# DeepFilterNet3 엔진
# ------------------------------------------------------------------
_df_state = None
_df_model = None


def _get_deepfilter():
    """DeepFilterNet 모델을 싱글톤으로 초기화."""
    global _df_state, _df_model
    if _df_state is None:
        from df.enhance import init_df
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state


def release_model() -> None:
    """DeepFilterNet 모델을 메모리에서 해제한다."""
    global _df_state, _df_model
    _df_state = None
    _df_model = None


def _denoise_deepfilter(
    audio: np.ndarray, sr: int, preset: str
) -> np.ndarray:
    """DeepFilterNet3으로 노이즈 제거."""
    import torch
    from df.enhance import enhance

    model, state = _get_deepfilter()
    params = _DEEPFILTER_PARAMS.get(preset, _DEEPFILTER_PARAMS["normal"])

    # DeepFilterNet은 내부 SR(48kHz)을 사용하므로 리샘플링 필요
    df_sr = state.sr()

    if sr != df_sr:
        # scipy 리샘플링
        from scipy.signal import resample

        n_samples = int(len(audio) * df_sr / sr)
        audio_resampled = resample(audio, n_samples).astype(np.float32)
    else:
        audio_resampled = audio

    # torch tensor로 변환 (1, T)
    tensor = torch.from_numpy(audio_resampled).unsqueeze(0)

    # enhance
    enhanced = enhance(model, state, tensor, atten_lim_db=params["atten_lim_db"])

    # numpy로 변환
    if isinstance(enhanced, torch.Tensor):
        result = enhanced.squeeze().cpu().numpy()
    else:
        result = np.array(enhanced).squeeze()

    # 원래 SR로 리샘플링
    if sr != df_sr:
        n_samples_out = int(len(result) * sr / df_sr)
        result = resample(result, n_samples_out).astype(np.float32)

    return result.astype(np.float32)


# ------------------------------------------------------------------
# noisereduce 폴백 엔진
# ------------------------------------------------------------------
def _denoise_noisereduce(
    audio: np.ndarray, sr: int, preset: str
) -> np.ndarray:
    """noisereduce로 노이즈 제거 (폴백)."""
    import noisereduce as nr

    params = _NOISEREDUCE_PARAMS.get(preset, _NOISEREDUCE_PARAMS["normal"])

    result = nr.reduce_noise(
        y=audio,
        sr=sr,
        stationary=params["stationary"],
        prop_decrease=params["prop_decrease"],
    )
    return result.astype(np.float32)


# ------------------------------------------------------------------
# 통합 인터페이스
# ------------------------------------------------------------------
def process(
    audio: np.ndarray,
    sr: int,
    preset: str = "normal",
    engine: str = "auto",
    **kwargs,
) -> np.ndarray:
    """노이즈 제거를 적용한다.

    Args:
        audio: float32 mono 오디오.
        sr: 샘플레이트 (16000).
        preset: "light" | "normal" | "strong".
        engine: "auto" (DeepFilterNet→noisereduce 폴백) | "lightweight" (noisereduce만, 메모리 절약).

    Returns:
        노이즈 제거된 오디오 (float32).
    """
    if preset == "light":
        logger.info("denoiser: light 프리셋 — 스킵")
        return audio

    engine_used = "none"
    has_df = engine == "auto" and _check_deepfilter()
    has_nr = _check_noisereduce()

    if has_df:
        try:
            audio = _denoise_deepfilter(audio, sr, preset)
            engine_used = "DeepFilterNet3"
        except Exception as exc:
            logger.warning("DeepFilterNet3 실패, noisereduce로 폴백: %s", exc)
            if has_nr:
                audio = _denoise_noisereduce(audio, sr, preset)
                engine_used = "noisereduce (fallback)"
            else:
                warnings.warn(
                    "DeepFilterNet3 실패 + noisereduce 미설치. 노이즈 제거 건너뜀.",
                    stacklevel=2,
                )
                engine_used = "none (both failed)"
    elif has_nr:
        audio = _denoise_noisereduce(audio, sr, preset)
        engine_used = f"noisereduce{' (lightweight)' if engine == 'lightweight' else ''}"
    else:
        warnings.warn(
            "노이즈 제거 엔진이 설치되지 않았습니다. "
            "설치: pip install deepfilternet 또는 pip install noisereduce",
            stacklevel=2,
        )
        engine_used = "none (not installed)"

    logger.info("denoiser: engine=%s, preset=%s", engine_used, preset)

    # strong 프리셋: 하이패스 필터로 키보드 저주파 제거
    if preset == "strong":
        audio = _highpass(audio, sr, cutoff=80.0)
        logger.info("denoiser: highpass 80Hz 적용")

    return audio
