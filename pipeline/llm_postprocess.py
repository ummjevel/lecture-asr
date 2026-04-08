"""LLM 후처리 — Claude API를 사용한 ASR 오류 교정, ITN, 단락 분리, 요약."""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Callable

from pipeline.models import Segment, TranscriptionResult

ProgressCallback = Callable[[str, float, str], None]

# 세그먼트 분할 기준: 약 10분 분량씩 LLM에 전달
SEGMENT_WINDOW_SEC = 10 * 60  # 10분

# ---------------------------------------------------------------------------
# 시스템 프롬프트
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
당신은 강의 음성 전사 교정 전문가입니다. ASR(자동 음성 인식) 출력을 교정합니다.

## 지시사항

1. **ASR 오류 교정**: 동음이의어, 전문용어 오인식을 문맥에 맞게 수정하세요.
   - 의미 없는 단어는 문맥에 맞는 올바른 단어로 교정
   - 전공 용어가 잘못 인식된 경우 올바른 표기로 수정

2. **ITN (Inverse Text Normalization)**:
   - 구어체 숫자를 아라비아 숫자로 변환 ("삼백이십오" → "325")
   - 날짜, 시간 표현 정규화 ("이천이십오년 사월 팔일" → "2025년 4월 8일")

3. **문장부호 복원**: 적절한 위치에 마침표, 쉼표, 물음표를 삽입하세요.

4. **필러 제거**: "음...", "어...", "그...", "아..." 등 불필요한 필러를 제거하세요.

5. **논리적 단락 분리**: 토픽이 전환되는 지점에서 빈 줄로 단락을 구분하세요.

## 출력 형식

교정된 텍스트만 출력하세요. 설명이나 주석은 포함하지 마세요.
원본의 의미와 내용을 변경하지 마세요. 오직 표현과 형식만 교정합니다."""

SUMMARY_PROMPT = """\
다음은 강의 전사 텍스트입니다. 핵심 내용을 요약하세요.

## 요약 지시사항

- 주요 토픽과 핵심 내용을 항목별로 정리
- 중요한 개념, 용어, 수식이 있으면 포함
- 분량: 원문의 10~20% 수준
- 한국어로 작성

## 출력 형식

# 강의 요약

## 주요 내용
- ...

## 핵심 개념
- ..."""


# ---------------------------------------------------------------------------
# Claude API 호출
# ---------------------------------------------------------------------------


def _get_client():
    """anthropic 클라이언트 생성."""
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "anthropic 라이브러리가 설치되지 않았습니다. "
            "pip install anthropic 으로 설치하세요."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY 환경변수가 설정되지 않았습니다. "
            "export ANTHROPIC_API_KEY='sk-...' 로 설정하세요."
        )

    return anthropic.Anthropic(api_key=api_key)


def _call_claude(
    client,
    system: str,
    user_message: str,
    max_tokens: int = 8192,
) -> str:
    """Claude API 단일 호출."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# 세그먼트 분할 (10분 윈도우)
# ---------------------------------------------------------------------------


def _split_into_windows(
    result: TranscriptionResult,
) -> list[str]:
    """전사 결과를 시간 윈도우 기반으로 텍스트 청크 분할."""
    if not result.segments:
        # 세그먼트 없으면 글자수 기준 분할 (약 3000자)
        text = result.text
        chunks: list[str] = []
        chunk_size = 3000
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks if chunks else [text]

    windows: list[str] = []
    current_texts: list[str] = []
    window_start = result.segments[0].start

    for seg in result.segments:
        if seg.start - window_start >= SEGMENT_WINDOW_SEC and current_texts:
            windows.append(" ".join(current_texts))
            current_texts = []
            window_start = seg.start
        current_texts.append(seg.text)

    if current_texts:
        windows.append(" ".join(current_texts))

    return windows


# ---------------------------------------------------------------------------
# 메인 후처리 함수
# ---------------------------------------------------------------------------


def postprocess_with_llm(
    result: TranscriptionResult,
    summary: bool = False,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """Claude API로 ASR 출력을 교정.

    Args:
        result: ASR 전사 결과
        summary: True 시 요약도 생성 (metadata["summary"]에 저장)
        on_progress: 진행 콜백 (step, percent, message)

    Returns:
        교정된 TranscriptionResult. 실패 시 원본 그대로 반환.
    """
    if on_progress:
        on_progress("llm_postprocess", 0.0, "LLM 후처리 시작...")

    try:
        client = _get_client()
    except (ImportError, ValueError) as e:
        if on_progress:
            on_progress("llm_postprocess", 100.0, f"LLM 후처리 스킵: {e}")
        return result

    # 원본 복사 (원본 보존)
    corrected = copy.deepcopy(result)

    try:
        # 텍스트 윈도우 분할
        windows = _split_into_windows(result)
        total = len(windows)
        corrected_parts: list[str] = []

        for idx, chunk in enumerate(windows):
            if on_progress:
                pct = (idx / total) * 80  # 교정: 0~80%
                on_progress(
                    "llm_postprocess",
                    pct,
                    f"교정 중... ({idx + 1}/{total})",
                )

            corrected_text = _call_claude(
                client,
                system=SYSTEM_PROMPT,
                user_message=chunk,
            )
            corrected_parts.append(corrected_text)

        corrected.text = "\n\n".join(corrected_parts)

        # 세그먼트 텍스트도 교정 결과 반영 (전체 텍스트 기준 재매핑)
        # 타임스탬프는 유지하되 텍스트만 업데이트
        if corrected.segments:
            _remap_segments(corrected)

        corrected.metadata["llm_corrected"] = True

        # 요약 생성
        if summary:
            if on_progress:
                on_progress("llm_postprocess", 85.0, "요약 생성 중...")

            summary_text = _call_claude(
                client,
                system=SUMMARY_PROMPT,
                user_message=corrected.text,
                max_tokens=4096,
            )
            corrected.metadata["summary"] = summary_text

        if on_progress:
            on_progress("llm_postprocess", 100.0, "LLM 후처리 완료")

        return corrected

    except Exception as e:
        # API 실패 시 원본 반환
        if on_progress:
            on_progress("llm_postprocess", 100.0, f"LLM 후처리 실패 (원본 유지): {e}")
        return result


def _remap_segments(result: TranscriptionResult) -> None:
    """교정된 전체 텍스트를 기존 세그먼트 타임스탬프에 비례 배분."""
    if not result.segments or not result.text:
        return

    full_text = result.text.replace("\n\n", " ").replace("\n", " ")
    total_original_len = sum(len(seg.text) for seg in result.segments)

    if total_original_len == 0:
        return

    # 각 세그먼트의 원본 비율에 따라 교정 텍스트 분배
    pos = 0
    for seg in result.segments:
        ratio = len(seg.text) / total_original_len
        char_count = max(1, int(len(full_text) * ratio))
        seg.text = full_text[pos:pos + char_count].strip()
        pos += char_count

    # 마지막 세그먼트에 남은 텍스트 할당
    if pos < len(full_text) and result.segments:
        remaining = full_text[pos:].strip()
        if remaining:
            result.segments[-1].text += " " + remaining
