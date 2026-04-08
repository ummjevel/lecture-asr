"""교차검증 — Whisper large-v3 (mlx-audio) 로 ASR 결과 비교."""

from __future__ import annotations

import difflib
from typing import Callable

from pipeline.models import Segment, TranscriptionResult

ProgressCallback = Callable[[str, float, str], None]

WHISPER_MODEL = "mlx-community/whisper-large-v3"


def _transcribe_whisper(
    audio_path: str,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """mlx-audio Whisper large-v3 로 전사."""
    try:
        from mlx_audio.whisper import transcribe as whisper_transcribe
    except ImportError:
        raise ImportError(
            "mlx-audio가 설치되지 않았습니다. "
            "pip install mlx-audio 로 설치하세요."
        )

    if on_progress:
        on_progress("cross_validate", 10.0, "Whisper 모델 로딩 중...")

    raw = whisper_transcribe(audio_path, model=WHISPER_MODEL, language="ko")

    if on_progress:
        on_progress("cross_validate", 80.0, "Whisper 전사 완료")

    # 결과 변환
    segments: list[Segment] = []
    if isinstance(raw, dict) and "segments" in raw:
        for seg in raw["segments"]:
            segments.append(Segment(
                start=seg.get("start", 0.0),
                end=seg.get("end", 0.0),
                text=seg.get("text", "").strip(),
            ))

    full_text = (
        raw.get("text", "") if isinstance(raw, dict)
        else " ".join(s.text for s in segments)
    )

    duration = 0.0
    if segments:
        duration = max(s.end for s in segments)

    return TranscriptionResult(
        text=full_text.strip(),
        segments=segments,
        language="ko",
        model=WHISPER_MODEL,
        duration=duration,
    )


def _build_segment_lines(result: TranscriptionResult) -> list[str]:
    """세그먼트를 비교 가능한 텍스트 라인 리스트로 변환."""
    if result.segments:
        return [seg.text for seg in result.segments]
    # 세그먼트가 없으면 문장 단위 분할
    import re
    sentences = re.split(r'(?<=[.?!。？！])\s+', result.text)
    return [s.strip() for s in sentences if s.strip()]


def _generate_diff_report(
    primary: TranscriptionResult,
    secondary: TranscriptionResult,
) -> str:
    """두 전사 결과를 비교하여 diff 리포트 생성."""
    lines_primary = _build_segment_lines(primary)
    lines_secondary = _build_segment_lines(secondary)

    diff = difflib.unified_diff(
        lines_primary,
        lines_secondary,
        fromfile=f"Primary ({primary.model})",
        tofile=f"Secondary ({secondary.model})",
        lineterm="",
    )
    diff_text = "\n".join(diff)

    # 유사도 계산
    matcher = difflib.SequenceMatcher(
        None,
        " ".join(lines_primary),
        " ".join(lines_secondary),
    )
    similarity = matcher.ratio() * 100

    report_parts = [
        "=" * 60,
        "교차검증 리포트 (Cross-Validation Report)",
        "=" * 60,
        "",
        f"Primary 모델:   {primary.model}",
        f"Secondary 모델: {secondary.model}",
        f"Primary 세그먼트 수:   {len(lines_primary)}",
        f"Secondary 세그먼트 수: {len(lines_secondary)}",
        f"텍스트 유사도: {similarity:.1f}%",
        "",
        "-" * 60,
        "차이점 (Unified Diff)",
        "-" * 60,
    ]

    if diff_text:
        report_parts.append(diff_text)
    else:
        report_parts.append("(차이 없음 — 두 모델 결과가 동일합니다)")

    report_parts.extend(["", "=" * 60])
    return "\n".join(report_parts)


def cross_validate(
    audio_path: str,
    primary_result: TranscriptionResult,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Whisper large-v3로 교차검증하여 diff 리포트 생성.

    Args:
        audio_path: WAV 파일 경로
        primary_result: 1차 ASR 전사 결과
        on_progress: 진행 콜백 (step, percent, message)

    Returns:
        비교 리포트 텍스트. mlx-audio 미설치 시 경고 메시지.
    """
    if on_progress:
        on_progress("cross_validate", 0.0, "교차검증 시작...")

    try:
        secondary_result = _transcribe_whisper(audio_path, on_progress)
    except ImportError as e:
        msg = (
            "[교차검증 스킵] mlx-audio가 설치되지 않았습니다.\n"
            "교차검증을 사용하려면: pip install mlx-audio\n"
            f"상세: {e}"
        )
        if on_progress:
            on_progress("cross_validate", 100.0, "mlx-audio 미설치 — 스킵")
        return msg
    except Exception as e:
        msg = f"[교차검증 오류] Whisper 전사 중 오류 발생: {e}"
        if on_progress:
            on_progress("cross_validate", 100.0, f"오류: {e}")
        return msg

    if on_progress:
        on_progress("cross_validate", 90.0, "리포트 생성 중...")

    report = _generate_diff_report(primary_result, secondary_result)

    if on_progress:
        on_progress("cross_validate", 100.0, "교차검증 완료")

    return report
