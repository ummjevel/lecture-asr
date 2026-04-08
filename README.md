# lecture-asr

**강의실 음성 전사 CLI 도구**

강의 녹음 파일(m4a, mp3, wav)을 넣으면 깨끗한 텍스트와 SRT 자막을 만들어주는 도구입니다.
M1/M2/M3/M4 맥북에서 완전히 로컬로 동작하며, 인터넷 없이도 사용할 수 있습니다.

---

## 왜 만들었나요?

기존 클로바 STT로 강의 녹음을 전사하면 이런 문제가 있었습니다:

- 강의실 소음(에어컨, 키보드 타이핑, 웅성거림)에 취약한 인식 품질
- "들리는 대로" 전사하여 의미 없는 단어가 다수 포함
- 전처리를 직접 제어할 수 없어 노이즈 환경에서 속수무책

**lecture-asr**는 이 문제를 해결합니다:

1. **5단계 오디오 전처리** — 노이즈 제거, 역반향 제거, 볼륨 정규화를 직접 제어
2. **Context Biasing** — 전공 용어를 미리 알려줘서 "의미 없는 단어" 문제 해결
3. **LLM 후처리** — Claude API로 오류 교정, 숫자 변환, 단락 분리까지

---

## 미리보기

처리 중에는 터미널에서 이런 화면을 볼 수 있습니다:

```
╭─ 강의 녹음 전사 ─────────────────────────────────╮
│  📁 lecture.m4a (55m 33s, 16kHz, mono)           │
│  🔧 노이즈 제거: normal                           │
│  🧠 모델: Qwen3-ASR-0.6B (4-bit)                 │
╰──────────────────────────────────────────────────╯

  [1/8] 오디오 변환   ━━━━━━━━━━━━━━━━━━ 100%  0:03 ✅
  [2/8] SNR 측정      ━━━━━━━━━━━━━━━━━━ 100%  0:01 ✅
  [3/8] 역반향 제거   ━━━━━━━━━━━━━━━━━━ 100%  0:12 ✅
  [4/8] 노이즈 제거   ━━━━━━━━━━━━━━━━━━ 100%  0:45 ✅
  [5/8] 볼륨 정규화   ━━━━━━━━━━━━━━━━━━ 100%  0:02 ✅
  [6/8] 음성 전사     ━━━━━━━━━━━━━━━━━  67%  8:21 🔴
  [7/8] 후처리        ░░░░░░░░░░░░░░░░░░  --   대기
  [8/8] 저장          ░░░░░░░░░░░░░░░░░░  --   대기

       🟥🟥🟥
    🟥🟥🟥🟥🟥
    ⬛⬛🔘⬛⬛      ✦  ·
    ⬜⬜⬜⬜⬜         ✧
      ⬜⬜⬜

  전체: ━━━━━━━━━━━━━━━━━━━━━━━━━━━  67%  9:24 경과
```

완료되면 몬스터볼 중앙이 ✨로 바뀌며 "딸깍!" 하고 파티클이 터집니다.

---

## 설치

### 사전 요구사항

- **macOS** (Apple Silicon M1/M2/M3/M4)
- **Python 3.10 이상**
- **ffmpeg** (오디오 변환에 필요)

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/ummjevel/lecture-asr.git
cd lecture-asr

# 2. ffmpeg 설치 (아직 없다면)
brew install ffmpeg

# 3. 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate

# 4. 의존성 설치
pip install -r requirements.txt
```

### 선택적 패키지

필요한 기능에 따라 추가로 설치할 수 있습니다:

```bash
# Whisper 교차검증을 사용하려면 (--cross-validate 옵션)
pip install mlx-audio

# LLM 후처리를 사용하려면 (--llm 옵션)
pip install anthropic
export ANTHROPIC_API_KEY="your-api-key"
```

---

## 사용법

### 기본 사용

```bash
# 가장 간단한 사용법 — 텍스트 파일로 출력
python transcribe.py lecture.m4a

# SRT 자막 파일로 출력
python transcribe.py lecture.m4a --format srt

# 텍스트 + SRT 둘 다 출력
python transcribe.py lecture.m4a --format both
```

### 노이즈 제거 강도 조절

강의실 환경에 따라 노이즈 제거 강도를 선택할 수 있습니다:

```bash
# 마이크 사용 강의 (소음 적음) → light
python transcribe.py lecture.m4a --denoise light

# 일반 강의실 (기본값) → normal
python transcribe.py lecture.m4a --denoise normal

# 대형 강의실, 키보드 소음 심함 → strong
python transcribe.py lecture.m4a --denoise strong

# 소음 수준을 자동으로 감지해서 결정
python transcribe.py lecture.m4a --denoise auto
```

| 프리셋 | 역반향 제거 | 노이즈 제거 | 볼륨 정규화 | 추천 환경 |
|--------|-----------|-----------|-----------|---------|
| `light` | 스킵 | 경미 | 스킵 | 마이크 사용, 소형 강의실 |
| `normal` | 적용 | 표준 | 적용 | 일반 강의실 (기본값) |
| `strong` | 적용 | 최대 + 하이패스 | 적용 | 대형 강의실, 소음 심함 |
| `auto` | SNR 기반 자동 | SNR 기반 자동 | SNR 기반 자동 | 잘 모르겠을 때 |

### 전공 용어 지정 (Context Biasing)

ASR이 전공 용어를 잘 인식하도록 미리 알려줄 수 있습니다.
이 기능은 클로바에서 "의미 없는 단어"가 나오던 문제를 해결하는 핵심입니다:

```bash
python transcribe.py lecture.m4a --context "머신러닝,트랜스포머,어텐션,역전파"
```

### LLM 후처리 (Claude API)

ASR 결과를 Claude가 한 번 더 다듬어줍니다:

```bash
# 오류 교정 + 숫자 변환 + 단락 분리
python transcribe.py lecture.m4a --llm

# 위 + 강의 요약까지 생성
python transcribe.py lecture.m4a --llm --summary
```

LLM 후처리가 하는 일:
- **오류 교정** — 동음이의어, 전문용어 오인식 수정
- **ITN** — "삼백이십오" → "325", 날짜/시간 변환
- **단락 분리** — 토픽 변화 기준으로 문단 나누기
- **요약 생성** — `--summary` 옵션 시 핵심 내용 요약

> 참고: `ANTHROPIC_API_KEY` 환경변수가 필요합니다.
> LLM 후처리를 사용하면 기본 후처리(문장부호/필러 제거)는 LLM이 통합 수행하므로 자동 스킵됩니다.

### 교차검증

전사 품질이 의심될 때 Whisper large-v3로 비교 리포트를 생성합니다:

```bash
python transcribe.py lecture.m4a --cross-validate
```

### 디렉토리 일괄 처리

폴더 안의 오디오 파일을 한꺼번에 처리합니다:

```bash
python transcribe.py ./lectures/ --format both
```

### 전체 옵션 조합 예시

```bash
# 소음 심한 대형 강의실 녹음, 전공 용어 포함, LLM 교정 + 요약, 텍스트+SRT 출력
python transcribe.py lecture.m4a \
  --denoise strong \
  --context "머신러닝,트랜스포머,어텐션" \
  --llm --summary \
  --format both
```

---

## 출력 파일

처리가 끝나면 원본 파일과 같은 폴더에 결과가 저장됩니다:

```
lectures/
├── lecture.m4a              ← 원본 녹음
├── lecture.txt              ← 전사 텍스트
├── lecture.srt              ← 타임스탬프 자막 (--format srt/both)
├── lecture.summary.txt      ← 강의 요약 (--llm --summary)
└── lecture_report.txt       ← 교차검증 리포트 (--cross-validate)
```

---

## 파이프라인 구조

```
입력(m4a/mp3/wav)
  │
  ├─ 1. 오디오 변환     ffmpeg → 16kHz mono WAV + 볼륨 정규화
  ├─ 2. SNR 측정        WADA SNR → 프리셋 자동 결정 (auto 시)
  ├─ 3. 역반향 제거     nara_wpe (강의실 반향 제거)
  ├─ 4. 노이즈 제거     DeepFilterNet3 / noisereduce
  ├─ 5. 볼륨 정규화     pyagc (원거리 음량 보정)
  ├─ 6. 음성 전사       mlx-qwen3-asr (Qwen3-ASR-0.6B, 4-bit)
  ├─ 7. 후처리          LLM 교정 또는 문장부호/필러 제거
  └─ 8. 저장            .txt / .srt / .summary.txt
```

---

## 프로젝트 구조

```
lecture-asr/
├── transcribe.py           # CLI 진입점
├── pipeline/
│   ├── models.py           # 공유 데이터 모델 (TranscriptionResult, Segment)
│   ├── __init__.py         # 전처리 오케스트레이터 (run_preprocess)
│   ├── converter.py        # ffmpeg 오디오 변환/정규화
│   ├── snr.py              # WADA SNR 측정 → 프리셋 자동 선택
│   ├── dereverb.py         # nara_wpe 역반향 제거
│   ├── denoiser.py         # DeepFilterNet3 / noisereduce 노이즈 제거
│   ├── agc.py              # pyagc 동적 볼륨 정규화
│   ├── asr.py              # mlx-qwen3-asr 전사 엔진
│   ├── postprocess.py      # 기본 후처리 (문장부호, 필러 제거)
│   ├── llm_postprocess.py  # Claude API 후처리
│   └── cross_validate.py   # Whisper 교차검증
├── ui/
│   ├── progress.py         # rich 프로그레스바 + 결과 요약
│   ├── pokeball.py         # 몬스터볼 애니메이션
│   └── particles.py        # 파티클 시스템
├── requirements.txt
└── docs/
    └── design.md           # 상세 설계 문서
```

---

## 시스템 요구사항

| 항목 | 요구사항 |
|------|---------|
| OS | macOS (Apple Silicon) |
| Python | 3.10 이상 |
| RAM | 최소 8GB, 권장 16GB |
| 디스크 | 모델 다운로드용 ~1GB |
| ffmpeg | 시스템에 설치 필요 |

### 메모리 사용량

| 모드 | 피크 메모리 |
|------|-----------|
| 일반 전사 | ~2GB |
| + 교차검증 | ~4GB |
| + LLM 후처리 | API 호출이므로 추가 메모리 없음 |

> 첫 실행 시 모델을 자동으로 다운로드합니다. 이후에는 인터넷 없이 사용 가능합니다.

### 모델 캐시 디렉토리

다운로드된 모델은 프로젝트 내부의 `.cache/` 폴더에 저장됩니다.
시스템 홈 디렉토리를 오염시키지 않으며, 프로젝트 단위로 관리할 수 있습니다.

```
lecture-asr/
└── .cache/
    ├── huggingface/    ← ASR, Whisper, 문장부호 모델
    └── torch/          ← DeepFilterNet 등 PyTorch 모델
```

> 이미 `HF_HOME` 등 환경변수를 직접 설정한 경우, 해당 설정이 우선 적용됩니다.

---

## 제약사항

- **Apple Silicon 전용** — MLX 프레임워크가 Intel Mac에서는 동작하지 않습니다
- **한국어 기준 최적화** — 영어 혼용 강의도 지원하지만, 한국어에 맞춰져 있습니다
- **첫 실행 시 인터넷 필요** — 모델 다운로드 후에는 오프라인 사용 가능

---

## 라이선스

MIT
