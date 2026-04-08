"""공유 데이터 모델 — 모든 파이프라인 모듈이 이 파일을 참조한다."""

from dataclasses import dataclass, field


@dataclass
class Segment:
    """타임스탬프가 있는 전사 구간."""

    start: float  # 시작 시간 (초)
    end: float  # 끝 시간 (초)
    text: str  # 전사 텍스트


@dataclass
class TranscriptionResult:
    """ASR 전사 결과. 후처리 모듈도 동일 구조를 입출력한다."""

    text: str  # 전체 텍스트
    segments: list[Segment] = field(default_factory=list)
    language: str = "ko"
    model: str = ""
    duration: float = 0.0  # 오디오 길이 (초)
    metadata: dict = field(default_factory=dict)
