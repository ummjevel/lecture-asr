# 강의실 음성 전사 CLI 도구 (Lecture ASR Tool)

## 1. 배경 및 목적

### 문제
- 현재 클로바 STT로 강의 녹음을 전사하고 있으나 품질 불만족
  - 노이즈에 취약: 강의실 환경음에 의한 오인식
  - 의미 없는 단어 출력: "들리는 대로" 전사하여 말이 안 되는 단어 다수
- 강의실 환경 특수성: 교수님 마이크 사용 여부 불규칙, 원거리 녹음

### 목표
- M1 Pro 16GB 맥북에서 로컬 실행되는 강의 전사 도구
- 클로바 대비 노이즈 환경에서의 품질 개선 (전처리 직접 제어)
- 스마트폰 녹음(m4a) → 깨끗한 텍스트/SRT 자막 출력
- 1~2시간+ 강의 녹음 처리 가능


### 샘플 파일 분석 (04-02.m4a)

| 항목 | 값 |
|---|---|
| 코덱 | AAC-LC (mp4a) |
| 샘플레이트 | 16kHz (ASR 입력과 동일, 리샘플링 불필요) |
| 채널 | mono |
| 비트레이트 | 64kbps |
| 길이 | 55분 33초 (3,332초) |
| 파일 크기 | 26MB |
| 인코더 | Lavf58.76.100 |

> 스마트폰 녹음앱이 이미 16kHz mono로 녹음하고 있어 전처리 변환 부담이 최소화됨.
> ffmpeg 단계에서는 AAC→WAV 디코딩 + 볼륨 정규화만 수행하면 됨.

### 실제 소음 환경 (청취 확인)

| 소음 유형 | 특성 | 빈도 |
|---|---|---|
| 배경 소음 (에어컨, 웅성거림 등) | stationary (정상 소음) | 상시 |
| 키보드 타이핑 | impulsive (충격 소음) | 간헐적 |
| 원거리 반향 | reverberation | 상시 |

> 녹음 기기: 스마트폰 녹음앱 또는 노트북 내장 마이크 (혼용)
> 키보드 타이핑 등 충격 소음은 일반 stationary 노이즈 제거만으로 불충분 —
> non-stationary 모드 병행 필수, strong 프리셋에서는 추가 필터링 고려.

### 강의실 반향(Reverberation) 환경

강의실 크기에 따라 잔향 시간(RT60)이 크게 달라지며, 원거리 녹음 시 반향이 ASR 정확도를 크게 떨어뜨림.

| 강의실 규모 | RT60 추정 | 반향 영향 | 권장 프리셋 |
|---|---|---|---|
| 소형 (20~30명) | 0.3~0.5초 | 낮음 | light |
| 중형 (50~100명) | 0.5~1.0초 | 중간 | normal |
| 대형 (100명+) | 1.0~2.0초+ | 높음 | strong |

> 역반향(dereverberation) 전처리를 파이프라인에 추가하여 대응.
> 사용자가 강의실 크기를 모르더라도 프리셋(light/normal/strong)으로 간편 선택 가능.

---

## 2. 파이프라인 아키텍처

```
[입력: m4a/mp3/wav]
    │
    ▼
┌──────────────────────────┐
│ 1. 오디오 변환/정규화      │  ffmpeg: m4a→wav, 16kHz, mono, 볼륨 정규화
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 2. SNR 측정 → 프리셋 자동  │  WADA SNR 추정
│   (--denoise auto 시)     │  >20dB→light, 10~20dB→normal, <10dB→strong
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 3. 역반향 제거            │  nara_wpe (강의실 반향 제거)
│   (normal/strong 시)      │  소형 강의실(light)은 스킵
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 4. 노이즈 제거            │  DeepFilterNet3 (기본, 신경망)
│   프리셋: light/normal/   │  또는 noisereduce (폴백)
│           strong          │
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 5. AGC (동적 볼륨 정규화)  │  pyagc — 원거리 음량 감쇠 보정
│   (normal/strong 시)      │  노이즈 제거 후 배치 (증폭 방지)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 6. ASR 전사               │  mlx-qwen3-asr (Qwen3-ASR-0.6B, 4-bit)
│   - 내장 VAD              │  자동 묵음 스킵
│   - 20분 단위 청크 분할    │  2시간+ 강의 대응
│   - 타임스탬프 생성        │  SRT/VTT 출력용
│   - Context biasing       │  --context "전공용어,..." 지원
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 7. LLM 후처리 (선택적)     │  Claude API 또는 로컬 LLM
│   - ASR 오류 교정          │  동음이의어, 의미없는 단어 수정
│   - ITN (숫자 변환)        │  "삼백이십오" → "325"
│   - 단락 분리              │  토픽 기준 문단 나누기
│   - 강의 요약 (선택)       │  핵심 내용 요약 생성
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ 8. 기본 후처리             │
│   - 문장부호 복원          │  deepmultilingualpunctuation
│   - 필러 제거              │  음..., 어..., 그... 등
│   - 텍스트 정리            │  줄바꿈/공백 정규화
└──────────────────────────┘
    │
    ▼
[출력: .txt / .srt / .summary.txt]
```

> **참고:** 7번 LLM 후처리 사용 시, 8번의 문장부호 복원과 필러 제거는 LLM이 통합 처리하므로 스킵.
> LLM 미사용 시 8번이 기본 후처리로 동작.

---

## 3. 핵심 컴포넌트

### 3.1 전처리 (Pre-processing)

**오디오 변환 (ffmpeg)**
- 입력: m4a, aac, mp3, wav 등 주요 포맷
- 출력: 16kHz, mono, WAV
- 샘플 파일 기준 이미 16kHz mono이므로 리샘플링 불필요, AAC→WAV 디코딩만 수행
- 볼륨 정규화: loudnorm 필터 적용 (원거리 녹음의 저볼륨 보정)
- 구현: subprocess로 ffmpeg CLI 호출

**SNR 자동 측정 (WADA SNR)**
- 오디오의 신호 대 잡음비(SNR)를 자동 추정
- `--denoise auto` 옵션 시 활성화, 수동 프리셋 지정 시 스킵
- 임계값: SNR > 20dB → light, 10~20dB → normal, < 10dB → strong
- 순수 Python (NumPy/SciPy), 의존성 최소

**역반향 제거 (nara_wpe)**
- 강의실 반향(reverberation) 제거, 노이즈 제거 전에 적용
- `light` 프리셋에서는 스킵 (소형 강의실은 반향 미미)
- `normal`/`strong` 프리셋에서 자동 적용
- WPE(Weighted Prediction Error) 알고리즘 기반, CPU 연산

**노이즈 제거 (DeepFilterNet3 기본, noisereduce 폴백)**

기본 엔진: DeepFilterNet3 (신경망 기반)
- PESQ 3.17~3.5, noisereduce(spectral gating) 대비 0.5~1점 이상 우위
- 파라미터 ~100만으로 매우 가벼움, M1 Pro MPS 또는 CoreML로 구동
- 키보드 타이핑 등 충격 소음에도 효과적
- 설치: `pip install deepfilternet`

폴백 엔진: noisereduce (DeepFilterNet 설치 실패 시)
- stationary + non-stationary 모드 병행

3단계 프리셋:
  - `light`: 경미한 처리, 마이크 사용 강의
  - `normal`: 표준 처리 (기본값)
  - `strong`: 최대 강도 + highpass 필터(키보드 저주파 제거)

**AGC — 동적 볼륨 정규화 (pyagc)**
- 교수님 이동에 따른 마이크 거리 변화 → 볼륨 들쭉날쭉 보정
- 노이즈 제거 후에 배치 (노이즈 증폭 방지)
- `normal`/`strong` 프리셋에서 자동 적용, `light`에서는 스킵
- NumPy/SciPy 기반, 가벼움

### 3.2 ASR 엔진

**Primary: mlx-qwen3-asr**
- 모델: Qwen3-ASR-0.6B, 4-bit 양자화
- 메모리: ~0.5GB
- 기능: VAD 내장, 자동 청크 분할(20분), txt/srt/vtt 출력
- 언어: 기본 한국어(ko), 자동 감지 옵션 제공
- **Context biasing**: `--context` 파라미터에 전공 용어 전달 → ASR이 해당 단어 방향으로 편향
  - 예: `--context "머신러닝, 트랜스포머, 어텐션, 역전파"`
  - Qwen3-ASR의 공식 기능 (SFT 단계에서 학습됨)
  - 클로바에서 "의미 없는 단어" 문제의 핵심 해결책
- 설치: `pip install "mlx-qwen3-asr[aligner]"`

**Secondary (선택적 교차검증): mlx-audio + Whisper large-v3**
- 용도: 전사 품질이 의심될 때 2차 모델로 비교
- 모델: mlx-community/whisper-large-v3 (한국어 WER 5-8%, 검증된 모델)
- 메모리: ~1.2GB 추가
- 출력: diff 형태의 비교 리포트
- 설치: `pip install mlx-audio` (필요 시에만)

### 3.3 LLM 후처리 (선택적, --llm 옵션)

ASR 출력을 LLM에 넣어 **오류 교정 + ITN + 단락 분리 + 요약을 단일 프롬프트로 통합 처리**.
별도 라이브러리 여러 개보다 효율적이며, 클로바의 "의미 없는 단어" 문제를 근본적으로 해결.

**기능:**
- ASR 오류 교정: 동음이의어, 전문용어 오인식 수정
- ITN (Inverse Text Normalization): "삼백이십오" → "325", 날짜/시간 변환
- 논리적 단락 분리: 토픽 변화 기준으로 문단 나누기
- 강의 요약 생성 (선택): `--summary` 옵션 시 핵심 내용 요약

**엔진 옵션:**
- Claude API (권장): 정확도 최고, 토큰 비용 발생
- 로컬 LLM: M1 Pro에서는 메모리 한계로 비권장
- 세그먼트 단위 처리: 2시간 강의를 10~15분 단위로 분할하여 LLM에 전달

**LLM 사용 시 기본 후처리(문장부호, 필러 제거)는 LLM이 통합 수행하므로 스킵.**

### 3.4 기본 후처리 (LLM 미사용 시)

**문장부호 복원**
- 라이브러리: deepmultilingualpunctuation
- 한국어/영어 지원
- ASR 출력에 마침표, 쉼표, 물음표 등 자동 삽입

**필러 제거**
- 패턴 매칭으로 반복 필러 제거: "음...", "어...", "그...", "아..." 등
- 설정으로 제거 수준 조절 가능

**텍스트 정리**
- 연속 공백 정규화
- 문단 분리 (긴 묵음 기준)
- 줄바꿈 정리

---

## 4. CLI 인터페이스

### 4.1 사용법

```bash
# 기본 사용 (텍스트 출력)
python transcribe.py lecture.m4a

# SRT 자막 출력
python transcribe.py lecture.m4a --format srt

# 둘 다 출력
python transcribe.py lecture.m4a --format both

# 노이즈 제거 강도 조절
python transcribe.py lecture.m4a --denoise strong

# 노이즈 자동 감지 (SNR 기반)
python transcribe.py lecture.m4a --denoise auto

# 전공 용어 지정 (Context biasing)
python transcribe.py lecture.m4a --context "머신러닝, 트랜스포머, 어텐션"

# LLM 후처리 (오류 교정 + ITN + 단락 분리)
python transcribe.py lecture.m4a --llm

# LLM 후처리 + 요약 생성
python transcribe.py lecture.m4a --llm --summary

# 교차검증 모드 (Whisper large-v3)
python transcribe.py lecture.m4a --cross-validate

# 디렉토리 일괄 처리
python transcribe.py ./lectures/ --format both

# 언어 지정
python transcribe.py lecture.m4a --lang ko
```

### 4.2 출력 구조

```
lectures/
├── lecture.m4a              (원본)
├── lecture.txt              (깨끗한 텍스트)
├── lecture.srt              (타임스탬프 자막)
├── lecture.summary.txt      (--summary 시, 강의 요약)
└── lecture_report.txt       (--cross-validate 시, 비교 리포트)
```

### 4.3 CLI UX (rich 라이브러리)

**프로그레스 표시 (8단계):**
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
```

**결과 요약:**
```
  ╭─ 결과 요약 ──────────────────────────────╮
  │  ✅ 전사 완료 (12분 34초)                 │
  │  📝 lecture.txt  (2,847 단어)             │
  │  🎬 lecture.srt  (342 구간)               │
  ╰──────────────────────────────────────────╯
```

---

## 5. 몬스터볼 애니메이션

처리 중 터미널에 표시되는 몬스터볼 픽셀아트 애니메이션. `rich.live.Live`로 프레임 로테이션.

### 5.1 몬스터볼 픽셀아트 (5x7 컴팩트)

```
        🟥🟥🟥
     🟥🟥🟥🟥🟥
     ⬛⬛🔘⬛⬛
     ⬜⬜⬜⬜⬜
       ⬜⬜⬜
```

### 5.2 애니메이션 상태

**처리 중 — 좌우 흔들림 (4프레임 루프)**
- 프레임1: 중앙 (기본 위치)
- 프레임2: 오른쪽으로 1칸 기울어짐 + `◐`
- 프레임3: 중앙 복귀
- 프레임4: 왼쪽으로 1칸 기울어짐 + `◑`

**완료 — 딸깍! 연출**
- 중앙 버튼 `🔘` → `✨` 전환
- 파티클 폭발 효과

### 5.3 파티클 시스템

프레임마다 몬스터볼 주변에 파티클을 랜덤 배치.

**파티클 종류:** `·` `✦` `✧` `✨` `⋆` `★`

**상태별 밀도:**

| 상태 | 파티클 수 | 종류 | 속도 |
|---|---|---|---|
| 처리 중 | 2~3개 | `· ✦ ✧` | 느리게 떠다님 |
| 거의 완료 (80%+) | 4~5개 | `· ✦ ✧ ⋆` | 점점 빨라짐 |
| 완료 | 8~10개 | `✨ ★ ✦ ✧ ·` | 터지듯 퍼짐 |

**구현:** 몬스터볼 주변 그리드(약 15x9)에서 랜덤 좌표에 파티클 배치. 매 프레임(200ms 간격)마다 위치 갱신.

---

## 6. 의존성

### 필수

| 패키지 | 버전 | 용도 | 메모리 |
|---|---|---|---|
| mlx-qwen3-asr[aligner] | latest | ASR 엔진 | ~0.5GB (4-bit) |
| deepfilternet | latest | 노이즈 제거 (기본) | ~0.1GB |
| noisereduce | latest | 노이즈 제거 (폴백) | ~0.1GB |
| nara_wpe | latest | 역반향 제거 | ~0.1GB |
| pyagc | latest | 동적 볼륨 정규화 | 무시 가능 |
| deepmultilingualpunctuation | latest | 문장부호 복원 | ~0.3GB |
| rich | latest | CLI UX/애니메이션 | 무시 가능 |
| ffmpeg | system | 오디오 변환 | CLI 도구 |

### 선택

| 패키지 | 용도 | 메모리 |
|---|---|---|
| mlx-audio | Whisper 교차검증 (--cross-validate) | ~1.2GB |
| anthropic | Claude API LLM 후처리 (--llm) | API 호출 |

### 메모리 예산 (M1 Pro 16GB)

- 일반 모드 피크: ~2GB
- LLM 후처리: API 호출이므로 로컬 메모리 추가 없음
- 교차검증 모드 피크: ~4GB
- 충분한 여유 확보

---

## 7. 프로젝트 구조

```
lecture-asr/
├── transcribe.py           # CLI 진입점 (argparse + rich)
├── pipeline/
│   ├── __init__.py
│   ├── converter.py        # ffmpeg 오디오 변환/정규화
│   ├── snr.py              # WADA SNR 추정 → 프리셋 자동 선택
│   ├── dereverb.py         # nara_wpe 역반향 제거
│   ├── denoiser.py         # DeepFilterNet3 (기본) / noisereduce (폴백)
│   ├── agc.py              # pyagc 동적 볼륨 정규화
│   ├── asr.py              # mlx-qwen3-asr 전사 + context biasing
│   ├── llm_postprocess.py  # LLM 후처리 (오류교정, ITN, 단락, 요약)
│   ├── postprocess.py      # 기본 후처리 (문장부호, 필러 제거, 텍스트 정리)
│   └── cross_validate.py   # 교차검증 — Whisper large-v3
├── ui/
│   ├── __init__.py
│   ├── pokeball.py         # 몬스터볼 픽셀아트 + 애니메이션 프레임
│   ├── particles.py        # 파티클 시스템
│   └── progress.py         # 프로그레스바 + 결과 요약 패널
├── requirements.txt
└── README.md
```

---

## 8. 설치 및 실행

```bash
# 1. ffmpeg 설치
brew install ffmpeg

# 2. Python 환경 (3.10+)
python -m venv .venv && source .venv/bin/activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. 실행
python transcribe.py lecture.m4a
```

---

## 9. 제약사항 및 향후 고려

### 제약사항
- Apple Silicon (M1/M2/M3/M4) 전용 — MLX는 x86 미지원
- 오프라인 실행 (첫 모델 다운로드 시에만 인터넷 필요)
- 한국어 기준 최적화, 영어 혼용 강의도 지원

### 향후 확장 가능
- Gradio 웹 UI 추가
- 스피커 다이어리제이션 (pyannote)
- 실시간 스트리밍 전사
- 커스텀 사전 (전공 용어 보정)
