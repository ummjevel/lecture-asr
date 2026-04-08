---
name: qa-inspect
description: "lecture-asr 파이프라인의 통합 정합성을 검증하는 스킬. 모듈 간 인터페이스 교차 비교, import 정합성, 데이터 흐름 연결성, 설계 대비 구현 완전성을 검사. 빌드 완료 후 검증 시 사용."
---

# 파이프라인 통합 정합성 검증 가이드

모든 모듈 구현 완료 후, 모듈 간 경계면을 교차 비교하여 통합 정합성을 검증한다.

## 검증 순서

### Step 1: 파일 존재 확인

모든 필수 파일이 생성되었는지 확인:

```
lecture-asr/
├── transcribe.py
├── pipeline/
│   ├── __init__.py
│   ├── models.py           # TranscriptionResult, Segment
│   ├── converter.py
│   ├── snr.py
│   ├── dereverb.py
│   ├── denoiser.py
│   ├── agc.py
│   ├── asr.py
│   ├── cross_validate.py
│   ├── llm_postprocess.py
│   └── postprocess.py
├── ui/
│   ├── __init__.py
│   ├── pokeball.py
│   ├── particles.py
│   └── progress.py
└── requirements.txt
```

### Step 2: 데이터 흐름 교차 비교 (최우선)

양쪽 코드를 **동시에** 열어 비교한다:

| 경계면 | 생산자 | 소비자 | 검증 항목 |
|--------|--------|--------|----------|
| convert→snr | converter.py `convert()` 반환 | snr.py `estimate_snr()` 입력 | ndarray + sr 형식 일치 |
| snr→preset | snr.py `auto_preset()` 반환 | 모든 전처리 모듈의 preset 파라미터 | 문자열 "light"/"normal"/"strong" |
| dereverb→denoise | dereverb.py `process()` 반환 | denoiser.py `process()` 입력 | ndarray 형식, sr 동일 |
| denoise→agc | denoiser.py `process()` 반환 | agc.py `process()` 입력 | ndarray 형식 |
| agc→asr | agc.py `process()` 반환 | asr.py `transcribe()` 입력 | WAV 파일 경로 또는 ndarray |
| asr→postprocess | asr.py TranscriptionResult | postprocess.py 입력 | TranscriptionResult 클래스 import |
| asr→llm_postprocess | asr.py TranscriptionResult | llm_postprocess.py 입력 | 동일 |
| CLI→pipeline | transcribe.py import | 각 모듈 실제 경로 | import 경로 정확성 |
| pipeline→UI | on_progress 콜백 시그니처 | progress.py update_progress | (step, percent, message) 일치 |

### Step 3: import 정합성

모든 파일의 import 문을 검사:
- `from pipeline.models import TranscriptionResult, Segment` — models.py에 실제 정의 존재?
- `from pipeline.converter import convert` — 함수명 정확?
- `import nara_wpe` — try/except로 감싸져 있는지? (선택 의존성)
- `import deepfilternet` — 폴백 로직 존재?

### Step 4: argparse ↔ 설계 문서 대조

docs/design.md의 CLI 사용법 섹션과 transcribe.py의 argparse를 1:1 대조:

```
설계서 옵션          → 구현 확인
--format txt/srt/both → ✅/❌
--denoise light/normal/strong/auto → ✅/❌
--context "용어,..." → ✅/❌
--llm                → ✅/❌
--summary            → ✅/❌
--cross-validate     → ✅/❌
--lang               → ✅/❌
```

### Step 5: 프리셋 일관성

3단계 프리셋(light/normal/strong)이 모든 전처리 모듈에서 동일하게 해석되는지:

| 모듈 | light | normal | strong |
|------|-------|--------|--------|
| dereverb | 스킵 | 적용 | 적용 |
| denoiser | 경미 | 표준 | 최대+highpass |
| agc | 스킵 | 적용 | 적용 |

### Step 6: requirements.txt 완전성

코드에서 import하는 모든 외부 패키지가 requirements.txt에 포함되어 있는지:

```
필수: mlx-qwen3-asr, noisereduce, nara-wpe, deepmultilingualpunctuation, rich, soundfile, numpy
선택: deepfilternet, pyagc, mlx-audio, anthropic
```

## 리포트 형식

`_workspace/qa_report.md`에 결과를 저장:

```markdown
# QA 검증 리포트

## 요약
- 전체: X/Y 통과
- CRITICAL: N건
- WARNING: N건

## 상세

### [CRITICAL] 모듈 간 인터페이스 불일치
- 위치: denoiser.py:45 ↔ agc.py:12
- 문제: denoiser가 tuple(audio, sr) 반환, agc가 ndarray만 기대
- 수정: agc.py 입력을 tuple로 변경하거나 denoiser 반환값 통일

### [WARNING] 설계 미구현
- --summary 옵션이 argparse에 없음
...
```
