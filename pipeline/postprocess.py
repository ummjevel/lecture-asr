"""기본 후처리 — 문장부호 복원, 필러 제거, 텍스트 정리."""

from __future__ import annotations

import copy
import re
from typing import Callable

from pipeline.models import Segment, TranscriptionResult

ProgressCallback = Callable[[str, float, str], None]

# ---------------------------------------------------------------------------
# 필러 패턴 (한국어 구어체)
# ---------------------------------------------------------------------------

FILLER_PATTERNS: list[re.Pattern] = [
    # "음...", "어...", "그...", "아..." 등 (점 개수 유동)
    re.compile(r'\b[음어그아으]{1,2}[.…·]{1,5}\s*', re.UNICODE),
    # "음, ", "어, ", "그, " (쉼표 뒤)
    re.compile(r'\b[음어그아으]{1,2},\s+', re.UNICODE),
    # 반복 필러: "그 그 그", "어 어"
    re.compile(r'\b([음어그아으])\s+\1(?:\s+\1)*\s*', re.UNICODE),
    # "그래서 이제", "뭐 이제" 같은 습관적 연결어 (단독 제거는 위험 — 최소한만)
    re.compile(r'\b뭐\s+이제\s+', re.UNICODE),
    # "에..." (단독)
    re.compile(r'\b에[.…]{1,5}\s*', re.UNICODE),
]


def _remove_fillers(text: str) -> str:
    """필러 패턴을 텍스트에서 제거."""
    for pat in FILLER_PATTERNS:
        text = pat.sub(" ", text)
    return text


def _remove_hallucination(text: str) -> str:
    """Whisper hallucination (동일 단어/구 반복, 깨진 문자) 제거."""
    # 같은 단어가 3회 이상 연속 반복되면 1회로
    text = re.sub(r'(\S+)(\s+\1){2,}', r'\1', text)
    # 같은 2~6어절 구가 2회 이상 반복되면 1회로
    text = re.sub(r'((?:\S+\s+){1,6}\S+)(?:\s+\1){1,}', r'\1', text)
    # 같은 한글 글자가 5회 이상 연속이면 제거 (예: "에에에에에")
    text = re.sub(r'([가-힣])\1{4,}', '', text)
    # 깨진 유니코드 문자 제거 (replacement character 등)
    text = re.sub(r'[�\ufffd]+', '', text)
    # <|th|> 같은 Whisper 토큰 잔여물 제거
    text = re.sub(r'<\|[^|]*\|>', '', text)
    # "밖에 없거든요" 류의 반복 문장 패턴 (같은 문장이 .으로 끝나고 반복)
    text = re.sub(r'((?:[^.!?]+[.!?])\s*)\1{1,}', r'\1', text)
    return text


# ---------------------------------------------------------------------------
# 문장부호 복원 (deepmultilingualpunctuation)
# ---------------------------------------------------------------------------

_punct_pipe = None
_PUNCT_MODEL_NAME = "oliverguhr/fullstop-punctuation-multilang-large"


def _get_punct_pipe():
    """문장부호 복원 pipeline lazy 로드 (transformers 5.x 호환)."""
    global _punct_pipe
    if _punct_pipe is not None:
        return _punct_pipe
    try:
        import logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        from transformers import pipeline as hf_pipeline
        _punct_pipe = hf_pipeline(
            "token-classification",
            model=_PUNCT_MODEL_NAME,
            aggregation_strategy="none",
        )
        return _punct_pipe
    except Exception:
        return None


def _restore_punctuation(text: str) -> str:
    """문장부호 복원. 라이브러리 미설치 시 원본 반환."""
    pipe = _get_punct_pipe()
    if pipe is None or not text.strip():
        return text

    try:
        # 긴 텍스트는 chunk로 분할 (512 토큰 제한)
        words = text.split()
        chunk_size = 200  # 단어 기준
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]

        result_parts = []
        for chunk in chunks:
            chunk_text = " ".join(chunk)
            labeled = pipe(chunk_text)
            for item in labeled:
                word = item["word"].strip()
                label = item.get("entity", "O")
                if not word:
                    continue
                # label: 0 (no punct), PERIOD, COMMA, QUESTION, EXCLAMATION
                punct_map = {"PERIOD": ".", "COMMA": ",", "QUESTION": "?", "EXCLAMATION": "!"}
                punct = punct_map.get(label, "")
                result_parts.append(word + punct)

        return " ".join(result_parts)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# 텍스트 정리
# ---------------------------------------------------------------------------


def _normalize_whitespace(text: str) -> str:
    """연속 공백/줄바꿈 정규화."""
    # 3줄 이상 빈 줄 → 2줄로
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 연속 공백 → 단일 공백 (줄바꿈 제외)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # 줄 앞뒤 공백 제거
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def _clean_text(text: str) -> str:
    """최종 텍스트 클리닝."""
    # sentencepiece 토큰 구분자 제거
    text = text.replace("▁", " ")
    # 반복 구두점 정리: "..." → ".", "??" → "?"
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    # 공백 + 구두점 → 구두점
    text = re.sub(r'\s+([.,?!])', r'\1', text)
    # 구두점 뒤 공백 보장 (줄바꿈 제외)
    text = re.sub(r'([.,?!])([^\s\n.,?!])', r'\1 \2', text)
    return text


# ---------------------------------------------------------------------------
# 세그먼트별 후처리 적용
# ---------------------------------------------------------------------------


def _correct_spacing(text: str) -> str:
    """한국어 띄어쓰기 교정. kss 미설치 시 원본 반환."""
    try:
        import kss
        return kss.correct_spacing(text)
    except Exception:
        return text


def _process_segment_text(text: str, use_punct: bool = True) -> str:
    """단일 텍스트 블록에 전체 후처리 파이프라인 적용."""
    text = _remove_fillers(text)
    text = _clean_text(text)
    text = _correct_spacing(text)
    if use_punct:
        text = _restore_punctuation(text)
    text = _normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------


def postprocess(
    result: TranscriptionResult,
    on_progress: ProgressCallback | None = None,
) -> TranscriptionResult:
    """기본 후처리: 문장부호 복원 + 필러 제거 + 텍스트 정리.

    Args:
        result: ASR 전사 결과
        on_progress: 진행 콜백 (step, percent, message)

    Returns:
        후처리된 TranscriptionResult (원본은 변경하지 않음)
    """
    if on_progress:
        on_progress("postprocess", 0.0, "기본 후처리 시작...")

    processed = copy.deepcopy(result)

    # deepmultilingualpunctuation 사용 가능 여부 확인
    punct_available = _get_punct_pipe() is not None

    if on_progress:
        status = "문장부호 복원 포함" if punct_available else "문장부호 복원 스킵 (미설치)"
        on_progress("postprocess", 10.0, status)

    # 전체 텍스트 후처리
    processed.text = _process_segment_text(processed.text, use_punct=punct_available)

    # 세그먼트별 후처리
    total_segs = len(processed.segments)
    for idx, seg in enumerate(processed.segments):
        seg.text = _remove_fillers(seg.text)
        seg.text = _clean_text(seg.text)
        seg.text = _normalize_whitespace(seg.text)
        # 세그먼트 단위 문장부호 복원은 스킵 (짧은 텍스트에 비효율)

        if on_progress and total_segs > 0:
            pct = 10 + (idx + 1) / total_segs * 80
            on_progress("postprocess", pct, f"세그먼트 처리 중... ({idx + 1}/{total_segs})")

    # 빈 세그먼트 제거
    processed.segments = [s for s in processed.segments if s.text.strip()]

    processed.metadata["postprocessed"] = True

    if on_progress:
        on_progress("postprocess", 100.0, "기본 후처리 완료")

    return processed
