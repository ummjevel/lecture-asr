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

DEFAULT_MODEL = "mlx-community/Qwen3-ASR-0.6B-4bit"
CHUNK_DURATION = 20 * 60  # 20분 (초)


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

    # 모델 로드
    model, processor = lib.load_model(model_id)

    if on_progress:
        on_progress("transcribe", 5.0, "모델 로딩 완료. 전사 시작...")

    # generate 인자 구성
    generate_kwargs: dict = {
        "language": lang,
    }

    # context biasing — 전공 용어 전달
    if context:
        generate_kwargs["context"] = context

    # 전사 실행 (mlx-qwen3-asr 내장 청크 분할 + VAD)
    raw_result = lib.generate_transcription(
        model=model,
        processor=processor,
        audio=audio_path,
        **generate_kwargs,
    )

    if on_progress:
        on_progress("transcribe", 90.0, "전사 완료. 결과 정리 중...")

    # raw_result → TranscriptionResult 변환
    segments: list[Segment] = []
    if hasattr(raw_result, "segments"):
        for seg in raw_result.segments:
            segments.append(Segment(
                start=seg.get("start", seg.get("t0", 0.0)),
                end=seg.get("end", seg.get("t1", 0.0)),
                text=seg.get("text", "").strip(),
            ))
    elif isinstance(raw_result, dict) and "segments" in raw_result:
        for seg in raw_result["segments"]:
            segments.append(Segment(
                start=seg.get("start", seg.get("t0", 0.0)),
                end=seg.get("end", seg.get("t1", 0.0)),
                text=seg.get("text", "").strip(),
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
