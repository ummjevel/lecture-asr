# QA 검증 리포트

## 요약
- 전체: 14/17 통과
- CRITICAL: 1건
- WARNING: 2건
- INFO: 3건

---

## 상세

### [CRITICAL] progress.py STEP 키 "transcribe" vs asr.py 콜백 키 "asr" 불일치

- 위치: `ui/progress.py:45` (`("transcribe", "음성 전사")`) vs `pipeline/asr.py:71` (`on_progress("asr", 0.0, ...)`)
- 문제: progress.py의 STEPS 리스트는 ASR 단계의 키를 `"transcribe"`로 정의하고 있으나, asr.py의 `transcribe()` 함수는 on_progress 콜백에 `"asr"` 키를 전달한다. 이로 인해:
  1. `_PlainUI.update_progress()`에서 `STEP_KEYS.index("asr")` 호출 시 `ValueError` 런타임 에러 발생
  2. `_RichUI`에서 ASR 진행률이 프로그레스바에 반영되지 않음 (`_step_percent`에 `"asr"` 키로 저장되지만, `_overall_percent()`가 `STEP_KEYS`의 `"transcribe"` 키만 조회)
- asr.py에서 `"asr"` 키가 사용되는 줄: 71, 77, 97, 139 (총 4곳)
- 수정: **asr.py의 on_progress 호출에서 `"asr"`를 `"transcribe"`로 변경** (4곳). 또는 progress.py STEPS에서 `"transcribe"`를 `"asr"`로 변경하되, transcribe.py의 다른 참조도 일괄 수정 필요. asr.py 쪽을 수정하는 것이 영향 범위가 작음.

---

### [WARNING] transcribe.py에서 후처리 모듈에 on_progress 콜백 미전달

- 위치: `transcribe.py:199-203`
- 문제: `postprocess_with_llm(result, summary=args.summary)` 및 `postprocess(result)` 호출 시 `on_progress` 인자를 전달하지 않는다. 두 함수 모두 `on_progress: ProgressCallback | None = None` 파라미터를 갖고 있으며, 내부에서 세부 진행률을 보고하도록 구현되어 있다. 현재는 수동으로 `ui.update_progress("postprocess", 0, "시작")`과 `100, "완료"`만 보고하므로 LLM 후처리의 세부 진행률(윈도우별 교정 진행, 요약 생성 등)이 UI에 표시되지 않는다.
- 수정:
  ```python
  # transcribe.py:199-203 수정
  if args.llm:
      result = postprocess_with_llm(result, summary=args.summary, on_progress=ui.update_progress)
  else:
      result = postprocess(result, on_progress=ui.update_progress)
  ```
  단, 이 경우 수동 `ui.update_progress("postprocess", 0/100, ...)` 호출은 제거해야 중복 방지.

---

### [WARNING] requirements.txt에 numpy, scipy, soundfile 누락

- 위치: `requirements.txt`
- 문제: 코드에서 직접 import하는 필수 패키지 3개가 requirements.txt에 명시되지 않았다:
  - `numpy` -- `pipeline/__init__.py`, `converter.py`, `snr.py`, `dereverb.py`, `denoiser.py`, `agc.py`에서 `import numpy as np`
  - `soundfile` -- `pipeline/__init__.py:11`, `converter.py:12`에서 `import soundfile as sf`
  - `scipy` -- `denoiser.py:9`에서 `from scipy.signal import butter, sosfilt`, `denoiser.py:95`에서 `from scipy.signal import resample`
- 현실적으로 `mlx-qwen3-asr`나 `noisereduce`의 전이 의존성으로 설치될 가능성이 높지만, 독립적으로 `pip install -r requirements.txt`만 실행했을 때 누락될 수 있다. 명시적 선언이 안전.
- 수정: requirements.txt에 추가:
  ```
  numpy
  scipy
  soundfile
  ```

---

### [INFO] llm_postprocess.py에서 json, Segment import 미사용

- 위치: `pipeline/llm_postprocess.py:6` (`import json`), `pipeline/llm_postprocess.py:11` (`from pipeline.models import Segment, TranscriptionResult`)
- 문제: `json` 모듈은 파일 내 어디에서도 사용되지 않는다. `Segment`도 직접 사용되지 않는다 (TranscriptionResult만 사용). 런타임 에러는 아니지만 불필요한 import.
- 수정: `import json` 제거, `Segment` import 제거 (`from pipeline.models import TranscriptionResult`만 유지).

---

### [INFO] postprocess.py에서 Segment import 미사용

- 위치: `pipeline/postprocess.py:9` (`from pipeline.models import Segment, TranscriptionResult`)
- 문제: `Segment`를 직접 사용하지 않는다. `TranscriptionResult`만 사용.
- 수정: `from pipeline.models import TranscriptionResult`로 변경.

---

### [INFO] cross_validate.py에서 on_progress 콜백도 "cross_validate" 키 사용 -- progress.py STEPS에 미등록

- 위치: `pipeline/cross_validate.py:140` (`on_progress("cross_validate", 0.0, ...)`) vs `ui/progress.py:38-47` (STEPS 8단계 정의)
- 문제: cross_validate 단계는 STEPS 8단계에 포함되지 않으므로 `_PlainUI.update_progress()`에서 `STEP_KEYS.index("cross_validate")` 호출 시 `ValueError` 발생 가능. 다만 현재 `transcribe.py:215-219`에서 `cross_validate()` 호출 시 `on_progress`를 전달하지 않으므로 실제 문제가 발생하지는 않는다. 향후 on_progress를 전달하면 에러 발생.
- 수정: (1) cross_validate에 on_progress 전달 시 STEPS에 추가하거나, (2) `_PlainUI.update_progress()`의 `STEP_KEYS.index(step)` 호출을 try/except로 감싸기.

---

## 통과 항목 (14건)

### Step 1: 파일 존재 확인 -- PASS
모든 17개 필수 파일이 존재함 확인 완료.

### Step 2: 데이터 흐름 교차 비교

| 경계면 | 결과 | 비고 |
|--------|------|------|
| convert -> snr | PASS | `converter.convert()` -> `(np.ndarray, int)`, `snr.estimate_snr(audio, sr)` -> `(np.ndarray, int)` 일치 |
| snr -> preset | PASS | `auto_preset()` -> `"light"/"normal"/"strong"` 문자열 반환, `__init__.py:56`에서 `effective_preset`으로 전달 |
| dereverb -> denoise | PASS | 둘 다 `process(audio, sr, preset)` -> `np.ndarray` 인터페이스. `__init__.py:65-70`에서 순차 호출 |
| denoise -> agc | PASS | 동일 `process(audio, sr, preset)` -> `np.ndarray` 인터페이스 |
| agc -> asr | PASS | `__init__.py:78`에서 `(audio, sr)` 반환 -> `save_preprocessed()` 브릿지로 WAV 파일 생성 -> `asr.transcribe(wav_path)` 파일 경로 전달 |
| asr -> postprocess | PASS | `asr.transcribe()` -> `TranscriptionResult`, `postprocess.postprocess(result)` -> `TranscriptionResult` 입력 |
| asr -> llm_postprocess | PASS | `llm_postprocess.postprocess_with_llm(result)` -> `TranscriptionResult` 입력. `copy.deepcopy` 사용 |
| pipeline -> UI 콜백 | **FAIL** | progress.py step key "transcribe" vs asr.py "asr" 불일치 (CRITICAL #1 참조) |

### Step 3: import 정합성 -- PASS (일부 INFO)

| 모듈 | import | 결과 |
|------|--------|------|
| pipeline/__init__.py | `from pipeline import converter, snr, dereverb, denoiser, agc` | PASS |
| pipeline/asr.py | `from pipeline.models import Segment, TranscriptionResult` | PASS |
| pipeline/cross_validate.py | `from pipeline.models import Segment, TranscriptionResult` | PASS |
| pipeline/llm_postprocess.py | `from pipeline.models import Segment, TranscriptionResult` | PASS (Segment 미사용 -- INFO) |
| pipeline/postprocess.py | `from pipeline.models import Segment, TranscriptionResult` | PASS (Segment 미사용 -- INFO) |
| transcribe.py | `from pipeline import run_preprocess, save_preprocessed` | PASS (`__init__.py`에 정의) |
| transcribe.py | `from pipeline.asr import transcribe, save_result` | PASS |
| transcribe.py | `from ui.progress import TranscribeUI` | PASS |
| ui/__init__.py | `from ui.progress import TranscribeUI` | PASS |
| 선택적 의존성 | nara_wpe, deepfilternet, pyagc, anthropic, mlx-audio | PASS (모두 try/except + 폴백) |

### Step 4: argparse vs 설계 문서 -- PASS

| 설계서 옵션 | 구현 | 결과 |
|-------------|------|------|
| `--format txt/srt/both` | `transcribe.py:39` choices=["txt", "srt", "both"] | PASS |
| `--denoise light/normal/strong/auto` | `transcribe.py:44` choices=["light", "normal", "strong", "auto"] | PASS |
| `--context "용어,..."` | `transcribe.py:49` | PASS |
| `--llm` | `transcribe.py:53` action="store_true" | PASS |
| `--summary` | `transcribe.py:58` action="store_true" + `--llm` 필수 검증 | PASS |
| `--cross-validate` | `transcribe.py:63` action="store_true" | PASS |
| `--lang` | `transcribe.py:68` default="ko" | PASS |
| `-v/--verbose` | `transcribe.py:73` | PASS (설계서에 명시적 없으나 합리적 추가) |

### Step 5: 프리셋 일관성 -- PASS

| 모듈 | light | normal | strong |
|------|-------|--------|--------|
| dereverb.py | 스킵 (line 56) | taps=10, delay=3, iter=3 | taps=20, delay=2, iter=5 |
| denoiser.py | atten_lim=12 / stationary+0.5 | atten_lim=20 / non-stat+0.75 | atten_lim=30 / non-stat+1.0 + highpass 80Hz |
| agc.py | 스킵 (line 94) | target=-20dB, maxgain=20dB | target=-18dB, maxgain=30dB |

설계 문서 대비 정확히 일치.
