# lecture-asr

강의실 음성 전사 CLI 도구. M1 Pro 맥북에서 로컬 실행.

## 아키텍처

- `transcribe.py` — CLI 진입점 (argparse + rich)
- `pipeline/` — 전처리(converter, snr, dereverb, denoiser, agc) + ASR + 후처리
- `ui/` — 프로그레스바, 몬스터볼 애니메이션, 파티클
- `pipeline/models.py` — 공유 데이터 모델 (TranscriptionResult, Segment)

## 핵심 규칙

- 모든 전처리 모듈(converter 제외)은 `process(audio: np.ndarray, sr: int, preset: str, **kwargs) -> np.ndarray` 인터페이스
- on_progress 콜백은 전체 통일: `(step: str, percent: float, message: str) -> None`
- 전처리 출력(ndarray)→ASR 입력(파일경로) 변환은 `save_preprocessed()` 브릿지 사용
- 선택적 의존성(nara_wpe, deepfilternet, pyagc, mlx-audio, anthropic)은 반드시 try/except + 폴백

## 빌드

`lecture-asr-build` 스킬로 전체 빌드. 3개 빌더 에이전트 병렬 → QA 검증 → 수정 루프.

## 설계 문서

- `docs/design.md` — 전체 설계 (파이프라인, CLI, UX, 의존성)
