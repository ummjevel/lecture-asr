---
name: asr-build
description: "ASR 엔진(mlx-qwen3-asr)과 후처리(LLM 교정, 문장부호, 필러 제거) 모듈을 구현하는 스킬. 강의 음성 전사, 교차검증, 텍스트 교정/변환 모듈 빌드 시 사용."
---

# ASR 엔진 및 후처리 모듈 구현 가이드

`pipeline/` 디렉토리에 4개 모듈을 구현한다.

## 공통 데이터 모델

모든 ASR/후처리 모듈이 공유하는 데이터 구조:

```python
from dataclasses import dataclass, field

@dataclass
class Segment:
    start: float       # 시작 시간 (초)
    end: float         # 끝 시간 (초)
    text: str          # 전사 텍스트

@dataclass
class TranscriptionResult:
    text: str                          # 전체 텍스트
    segments: list[Segment] = field(default_factory=list)  # 타임스탬프 세그먼트
    language: str = "ko"               # 감지된 언어
    model: str = ""                    # 사용된 모델명
    duration: float = 0.0              # 오디오 길이 (초)
    metadata: dict = field(default_factory=dict)  # 추가 메타데이터
```

이 데이터 클래스를 `pipeline/models.py`에 정의하고 모든 모듈에서 import한다.

## 모듈별 구현 상세

### 1. asr.py — mlx-qwen3-asr 전사

```python
def transcribe(audio_path: str, lang: str = "ko", context: str | None = None,
               on_progress=None) -> TranscriptionResult:
    """
    mlx-qwen3-asr로 음성 전사.
    Args:
        audio_path: WAV 파일 경로
        lang: 언어 코드 (기본 "ko")
        context: 전공 용어 (쉼표 구분, context biasing용)
        on_progress: 콜백 (step: str, percent: float, message: str) -> None
                     전처리 모듈과 동일한 시그니처. ASR 내부에서 chunk 진행률을 percent로 변환하여 호출.
                     예: on_progress("음성 전사", chunk_idx/total_chunks, f"청크 {chunk_idx}/{total_chunks}")
    """
```

핵심 구현 사항:
- mlx-qwen3-asr 라이브러리의 `load_model`, `generate_transcription` 활용
- 4-bit 양자화 모델 사용 (`mlx-community/Qwen3-ASR-0.6B-bf16` 또는 양자화 버전)
- context biasing: `--context` 값을 모델의 context 파라미터에 전달
- 긴 오디오는 mlx-qwen3-asr 내장 청크 분할 활용 (20분 단위)
- 각 청크 완료 시 on_progress 콜백 호출
- SRT 형식 출력 지원: segments를 SRT 포맷으로 변환하는 헬퍼 함수 포함

```python
def to_srt(result: TranscriptionResult) -> str:
    """TranscriptionResult → SRT 형식 문자열"""

def save_result(result: TranscriptionResult, output_path: str, format: str = "txt"):
    """결과를 파일로 저장 (txt, srt, both)"""
```

### 2. cross_validate.py — Whisper 교차검증

```python
def cross_validate(audio_path: str, primary_result: TranscriptionResult,
                   on_progress=None) -> str:
    """
    Whisper large-v3로 교차검증하여 diff 리포트 생성.
    Returns: 비교 리포트 텍스트
    """
```

- mlx-audio의 Whisper large-v3 사용
- 두 결과를 세그먼트 단위로 비교하여 차이점 표시
- mlx-audio 미설치 시: 경고 메시지 반환 (에러 아님)

### 3. llm_postprocess.py — LLM 후처리

```python
def postprocess_with_llm(result: TranscriptionResult,
                         summary: bool = False) -> TranscriptionResult:
    """
    Claude API로 ASR 출력을 교정.
    - 오류 교정 (동음이의어, 전문용어)
    - ITN ("삼백이십오" → "325")
    - 논리적 단락 분리
    - 요약 생성 (summary=True 시)
    """
```

- anthropic 라이브러리 사용 (Claude API)
- 2시간 강의를 10~15분 세그먼트로 분할하여 LLM에 전달
- 시스템 프롬프트에 역할(강의 전사 교정) + 지시사항(ITN, 단락 분리) 포함
- API 키는 환경변수 `ANTHROPIC_API_KEY`에서 읽기
- API 실패 시 원본 result 그대로 반환 (폴백)
- summary=True 시 별도 요약 요청 → result.metadata["summary"]에 저장

### 4. postprocess.py — 기본 후처리

```python
def postprocess(result: TranscriptionResult) -> TranscriptionResult:
    """문장부호 복원 + 필러 제거 + 텍스트 정리"""
```

- deepmultilingualpunctuation으로 문장부호 복원
- 필러 패턴 제거: 정규식으로 "음...", "어...", "그...", "아..." 등
- 텍스트 정리: 연속 공백 정규화, 줄바꿈 정리
- deepmultilingualpunctuation 미설치 시 문장부호 복원 스킵

## 모듈 간 데이터 흐름

```
전처리된 WAV → asr.transcribe() → TranscriptionResult
                                        │
                            ┌────────────┴────────────┐
                            │                         │
                    --llm 사용 시              --llm 미사용 시
                            │                         │
              llm_postprocess.postprocess_with_llm()   postprocess.postprocess()
                            │                         │
                            └────────────┬────────────┘
                                        │
                              TranscriptionResult (교정됨)
                                        │
                              save_result() → .txt / .srt
```
