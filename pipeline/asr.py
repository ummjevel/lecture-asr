"""mlx-qwen3-asr 전사 엔진 — context biasing, SRT 변환, 결과 저장."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from pipeline.models import Segment, TranscriptionResult

# on_progress 콜백 시그니처: (step: str, percent: float, message: str) -> None
ProgressCallback = Callable[[str, float, str], None]

# ---------------------------------------------------------------------------
# mlx-qwen3-asr lazy import
# ---------------------------------------------------------------------------

_mlx_asr = None


def _load_mlx_asr():
    global _mlx_asr
    if _mlx_asr is not None:
        return _mlx_asr
    try:
        import mlx_qwen3_asr  # noqa: F811
        _mlx_asr = mlx_qwen3_asr
        return _mlx_asr
    except ImportError:
        raise ImportError(
            "mlx-qwen3-asr가 설치되지 않았습니다. "
            "pip install 'mlx-qwen3-asr[aligner]' 로 설치하세요."
        )


# ---------------------------------------------------------------------------
# 기본 모델 설정
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
FORCED_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
CHUNK_DURATION = 20 * 60  # 20분 (초)


def _patched_load_model(model_id: str, dtype=None):
    """lm_head.weight 누락 문제를 우회하여 모델을 로드한다."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_qwen3_asr.load_models import (
        _resolve_path, _load_safetensors, remap_weights,
        _read_quantization_config, _is_quantized_weights,
    )
    from mlx_qwen3_asr.config import Qwen3ASRConfig
    from mlx_qwen3_asr.model import Qwen3ASRModel
    import json

    if dtype is None:
        dtype = mx.bfloat16

    model_path = _resolve_path(model_id)

    with open(model_path / "config.json") as f:
        raw_config = json.load(f)
    config = Qwen3ASRConfig.from_dict(raw_config)

    weights = _load_safetensors(model_path)
    weights = remap_weights(weights)

    # lm_head.weight 누락 시 embed_tokens에서 복사
    if "lm_head.weight" not in weights and "model.embed_tokens.weight" in weights:
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

    model = Qwen3ASRModel(config)

    quant_cfg = _read_quantization_config(model_path)
    quantized = _is_quantized_weights(weights)
    if quantized:
        bits = int(quant_cfg.get("bits", 4)) if quant_cfg else 4
        group_size = int(quant_cfg.get("group_size", 64)) if quant_cfg else 64
        nn.quantize(model, bits=bits, group_size=group_size)

    model.load_weights(list(weights.items()))

    if dtype != mx.float32 and not quantized:
        import mlx.utils as mlx_utils
        def _cast(x):
            if isinstance(x, mx.array) and x.dtype in (mx.float32, mx.float16, mx.bfloat16):
                return x.astype(dtype)
            return x
        params = mlx_utils.tree_map(_cast, model.parameters())
        model.load_weights(list(mlx_utils.tree_flatten(params)))

    mx.eval(model.parameters())
    model.eval()
    setattr(model, "_source_model_id", model_id)
    setattr(model, "_resolved_model_path", str(model_path))

    return model, config, model_path


# ---------------------------------------------------------------------------
# 핵심 전사 함수
# ---------------------------------------------------------------------------


def transcribe(
    audio_path: str,
    lang: str = "ko",
    context: str | None = None,
    on_progress: ProgressCallback | None = None,
    model_id: str = DEFAULT_MODEL,
) -> TranscriptionResult:
    """mlx-qwen3-asr로 음성 전사.

    Args:
        audio_path: WAV 파일 경로 (전처리 완료된 16kHz mono)
        lang: 언어 코드 (기본 "ko")
        context: 전공 용어 쉼표 구분 문자열 (context biasing용)
        on_progress: 진행 콜백 (step, percent, message)
        model_id: 모델 ID (기본 4-bit 양자화)

    Returns:
        TranscriptionResult
    """
    lib = _load_mlx_asr()

    if on_progress:
        on_progress("transcribe", 0.0, "모델 로딩 중...")

    # 패치된 로더로 모델 로드 (lm_head.weight 누락 우회)
    import mlx.core as mx
    loaded_model, _config, _ = _patched_load_model(model_id, dtype=mx.float16)

    if on_progress:
        on_progress("transcribe", 5.0, "모델 로딩 완료. 전사 시작...")

    # 라이브러리 on_progress → UI 프로그레스 변환
    def _asr_progress(event: dict) -> None:
        if not on_progress:
            return
        evt = event.get("event", "")
        if evt == "chunk_started":
            idx = event.get("chunk_index", 0)
            total = event.get("total_chunks", 1)
            pct = 5.0 + (idx / total) * 85.0  # 5%~90% 범위
            on_progress("transcribe", pct, f"전사 중... ({idx}/{total} 청크)")
        elif evt == "completed":
            on_progress("transcribe", 90.0, "전사 완료. 결과 정리 중...")

    # 전사 실행 (verbose=False로 stdout 누출 방지)
    # forced_aligner 로딩 실패 시 stdout/stderr 억제 후 타임스탬프 없이 재시도
    import io, sys
    try:
        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            raw_result = lib.transcribe(
                audio_path,
                model=loaded_model,
                language=lang,
                context=context or "",
                return_timestamps=True,
                forced_aligner=FORCED_ALIGNER_MODEL,
                verbose=False,
                on_progress=_asr_progress,
            )
        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
    except Exception:
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        if on_progress:
            on_progress("transcribe", 8.0, "Aligner 실패 — 타임스탬프 없이 재시도...")
        raw_result = lib.transcribe(
            audio_path,
            model=loaded_model,
            language=lang,
            context=context or "",
            return_timestamps=False,
            verbose=False,
            on_progress=_asr_progress,
        )

    if on_progress:
        on_progress("transcribe", 90.0, "전사 완료. 결과 정리 중...")

    # raw_result → TranscriptionResult 변환
    segments: list[Segment] = []
    raw_segments = None
    if hasattr(raw_result, "segments") and raw_result.segments:
        raw_segments = raw_result.segments
    elif isinstance(raw_result, dict) and raw_result.get("segments"):
        raw_segments = raw_result["segments"]

    if raw_segments:
        for seg in raw_segments:
            if isinstance(seg, dict):
                segments.append(Segment(
                    start=seg.get("start", seg.get("t0", 0.0)),
                    end=seg.get("end", seg.get("t1", 0.0)),
                    text=seg.get("text", "").strip(),
                ))
            else:
                segments.append(Segment(
                    start=getattr(seg, "start", getattr(seg, "t0", 0.0)),
                    end=getattr(seg, "end", getattr(seg, "t1", 0.0)),
                    text=getattr(seg, "text", "").strip(),
                ))

    full_text = (
        getattr(raw_result, "text", None)
        or (raw_result.get("text") if isinstance(raw_result, dict) else None)
        or " ".join(s.text for s in segments)
    )

    # 오디오 길이 추정
    duration = 0.0
    if segments:
        duration = max(s.end for s in segments)
    elif isinstance(raw_result, dict):
        duration = raw_result.get("duration", 0.0)

    result = TranscriptionResult(
        text=full_text.strip(),
        segments=segments,
        language=lang,
        model=model_id,
        duration=duration,
        metadata={"context": context} if context else {},
    )

    if on_progress:
        on_progress("transcribe", 100.0, f"전사 완료 ({len(segments)}개 세그먼트)")

    return result


# ---------------------------------------------------------------------------
# Whisper (mlx-whisper) 전사 엔진
# ---------------------------------------------------------------------------

WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"
KO_WHISPER_MODEL = "youngouk/whisper-medium-komixv2-mlx"


def _patch_tqdm(on_progress: ProgressCallback | None):
    """tqdm.tqdm을 monkey-patch하여 stderr 출력 억제 + on_progress 전달.

    contextmanager로 사용: with _patch_tqdm(cb): ...
    """
    import contextlib
    import tqdm as tqdm_mod

    @contextlib.contextmanager
    def _ctx():
        orig_tqdm = tqdm_mod.tqdm

        class _SilentTqdm(orig_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True  # stderr 출력 억제
                super().__init__(*args, **kwargs)
                self._total_frames = kwargs.get("total") or (args[0] if args else None)
                self._accumulated = 0

            def update(self, n=1):
                self._accumulated += n
                if on_progress and self._total_frames:
                    # 10%~90% 범위로 매핑
                    pct = 10.0 + (self._accumulated / self._total_frames) * 80.0
                    on_progress("transcribe", min(pct, 90.0),
                                f"전사 중... ({self._accumulated}/{self._total_frames} 프레임)")
                return super().update(n)

        tqdm_mod.tqdm = _SilentTqdm
        try:
            yield
        finally:
            tqdm_mod.tqdm = orig_tqdm

    return _ctx()


def transcribe_whisper(
    audio_path: str,
    lang: str = "ko",
    on_progress: ProgressCallback | None = None,
    model_id: str = WHISPER_MODEL,
) -> TranscriptionResult:
    """mlx-audio Whisper로 음성 전사.

    Args:
        audio_path: WAV 파일 경로
        lang: 언어 코드 (기본 "ko")
        on_progress: 진행 콜백
        model_id: Whisper 모델 ID

    Returns:
        TranscriptionResult
    """
    try:
        from mlx_audio.stt.utils import load_model
        from mlx_audio.stt.generate import generate_transcription
    except ImportError:
        raise ImportError(
            "mlx-audio가 설치되지 않았습니다. "
            "pip install mlx-audio 로 설치하세요."
        )

    if on_progress:
        on_progress("transcribe", 5.0, "Whisper 모델 로딩 중...")

    model = load_model(model_id)

    if on_progress:
        on_progress("transcribe", 10.0, "모델 로딩 완료. 전사 중...")

    with _patch_tqdm(on_progress):
        raw_result = generate_transcription(
            model=model,
            audio=audio_path,
            verbose=False,
            language=lang,
        )

    if on_progress:
        on_progress("transcribe", 95.0, "전사 완료. 결과 정리 중...")

    # raw_result → TranscriptionResult 변환
    segments: list[Segment] = []
    if hasattr(raw_result, "segments") and raw_result.segments:
        for seg in raw_result.segments:
            if isinstance(seg, dict):
                segments.append(Segment(
                    start=seg.get("start", 0.0),
                    end=seg.get("end", 0.0),
                    text=seg.get("text", "").strip(),
                ))
            else:
                segments.append(Segment(
                    start=getattr(seg, "start", 0.0),
                    end=getattr(seg, "end", 0.0),
                    text=getattr(seg, "text", "").strip(),
                ))

    full_text = (
        getattr(raw_result, "text", None)
        or " ".join(s.text for s in segments)
    )

    duration = 0.0
    if segments:
        duration = max(s.end for s in segments)

    result = TranscriptionResult(
        text=full_text.strip(),
        segments=segments,
        language=lang,
        model=model_id,
        duration=duration,
        metadata={},
    )

    if on_progress:
        on_progress("transcribe", 100.0, f"전사 완료 ({len(segments)}개 세그먼트)")

    return result


# ---------------------------------------------------------------------------
# 한국어 특화 Whisper (mlx_whisper) 전사 엔진
# ---------------------------------------------------------------------------


def transcribe_ko_whisper(
    audio_path: str,
    lang: str = "ko",
    on_progress: ProgressCallback | None = None,
    model_id: str = KO_WHISPER_MODEL,
) -> TranscriptionResult:
    """한국어 특화 Whisper (whisper-medium-komixv2-mlx)로 음성 전사.

    mlx_whisper 패키지를 사용한다 (mlx-audio와 별도).

    Args:
        audio_path: WAV 파일 경로
        lang: 언어 코드 (기본 "ko")
        on_progress: 진행 콜백
        model_id: 모델 ID

    Returns:
        TranscriptionResult
    """
    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx-whisper가 설치되지 않았습니다. "
            "pip install mlx-whisper 로 설치하세요."
        )

    import gc
    import soundfile as sf
    import tempfile

    if on_progress:
        on_progress("transcribe", 5.0, "한국어 Whisper 모델 로딩 중...")

    # 오디오 길이 확인 → 긴 파일은 청크 분할 (메모리 보호)
    info = sf.info(audio_path)
    total_duration = info.duration
    chunk_sec = 300  # 5분 단위

    if on_progress:
        on_progress("transcribe", 10.0, "모델 로딩 완료. 전사 중...")

    all_segments: list[Segment] = []
    all_texts: list[str] = []

    if total_duration <= chunk_sec + 30:
        # 짧은 오디오: 한번에 처리
        with _patch_tqdm(on_progress):
            raw_result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model_id,
                language=lang,
                verbose=False,
                word_timestamps=False,
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.0,
            )
        all_texts.append(raw_result.get("text", ""))
        for seg in raw_result.get("segments", []):
            all_segments.append(Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            ))
    else:
        # 긴 오디오: 5분 청크로 분할 처리 — 전체 로드 없이 파일에서 청크 단위로 읽는다
        import math
        sr = info.samplerate
        total_frames = info.frames
        chunk_frames = int(chunk_sec * sr)
        n_chunks = math.ceil(total_frames / chunk_frames)

        with sf.SoundFile(audio_path) as snd:
            for i in range(n_chunks):
                start_frame = i * chunk_frames
                end_frame = min(start_frame + chunk_frames, total_frames)
                time_offset = i * chunk_sec

                if on_progress:
                    pct = 10.0 + (i / n_chunks) * 80.0
                    on_progress("transcribe", pct,
                                f"전사 중... ({i + 1}/{n_chunks} 청크)")

                snd.seek(start_frame)
                chunk = snd.read(frames=end_frame - start_frame, dtype="float32", always_2d=False)
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)

                # 청크를 임시 WAV로 저장 — 생성 직후 try/finally로 보호
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False, prefix="ko-whisper-chunk-",
                )
                tmp_name = tmp.name
                tmp.close()
                chunk_result = None
                try:
                    sf.write(tmp_name, chunk, sr, subtype="PCM_16")
                    # chunk는 write 이후 더 이상 필요 없으므로 즉시 해제
                    del chunk
                    with _patch_tqdm(None):  # 청크별 tqdm 억제
                        chunk_result = mlx_whisper.transcribe(
                            tmp_name,
                            path_or_hf_repo=model_id,
                            language=lang,
                            verbose=False,
                            word_timestamps=False,
                            condition_on_previous_text=False,
                            hallucination_silence_threshold=2.0,
                            no_speech_threshold=0.6,
                            compression_ratio_threshold=2.0,
                        )
                finally:
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass

                if chunk_result is not None:
                    all_texts.append(chunk_result.get("text", ""))
                    for seg in chunk_result.get("segments", []):
                        all_segments.append(Segment(
                            start=seg.get("start", 0.0) + time_offset,
                            end=seg.get("end", 0.0) + time_offset,
                            text=seg.get("text", "").strip(),
                        ))

                # 청크 간 메모리 해제
                del chunk_result
                gc.collect()

    if on_progress:
        on_progress("transcribe", 95.0, "전사 완료. 결과 정리 중...")

    full_text = " ".join(all_texts).strip() or " ".join(s.text for s in all_segments)
    duration = max((s.end for s in all_segments), default=0.0)

    result = TranscriptionResult(
        text=full_text,
        segments=all_segments,
        language=lang,
        model=model_id,
        duration=duration,
        metadata={},
    )

    if on_progress:
        on_progress("transcribe", 100.0, f"전사 완료 ({len(all_segments)}개 세그먼트)")

    return result


# ---------------------------------------------------------------------------
# SRT 변환 / 저장 헬퍼
# ---------------------------------------------------------------------------


def _format_srt_time(seconds: float) -> str:
    """초 → SRT 타임코드 (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(result: TranscriptionResult) -> str:
    """TranscriptionResult -> SRT 형식 문자열."""
    if not result.segments:
        return ""

    lines: list[str] = []
    for idx, seg in enumerate(result.segments, 1):
        start_tc = _format_srt_time(seg.start)
        end_tc = _format_srt_time(seg.end)
        lines.append(str(idx))
        lines.append(f"{start_tc} --> {end_tc}")
        lines.append(seg.text)
        lines.append("")  # 빈 줄 구분

    return "\n".join(lines)


def save_result(
    result: TranscriptionResult,
    output_path: str,
    format: str = "txt",
) -> list[str]:
    """결과를 파일로 저장.

    Args:
        result: 전사 결과
        output_path: 출력 경로 (확장자 없이 또는 있어도 무관)
        format: "txt", "srt", "both"

    Returns:
        생성된 파일 경로 리스트
    """
    base = str(Path(output_path).with_suffix(""))
    created: list[str] = []

    if format in ("txt", "both"):
        txt_path = f"{base}.txt"
        Path(txt_path).write_text(result.text, encoding="utf-8")
        created.append(txt_path)

    if format in ("srt", "both"):
        srt_path = f"{base}.srt"
        srt_content = to_srt(result)
        Path(srt_path).write_text(srt_content, encoding="utf-8")
        created.append(srt_path)

    # 요약이 있으면 별도 파일
    summary = result.metadata.get("summary")
    if summary:
        summary_path = f"{base}.summary.txt"
        Path(summary_path).write_text(summary, encoding="utf-8")
        created.append(summary_path)

    return created
