---
name: preprocess-build
description: "오디오 전처리 파이프라인 모듈(converter, snr, dereverb, denoiser, agc)을 구현하는 스킬. 강의실 녹음의 노이즈 제거, 역반향 처리, 볼륨 정규화를 담당하는 Python 모듈 빌드 시 사용."
---

# 오디오 전처리 모듈 구현 가이드

`pipeline/` 디렉토리 하위에 5개 전처리 모듈을 구현한다. 모든 모듈은 동일한 패턴을 따른다.

## 공통 인터페이스

모든 전처리 모듈은 다음 패턴으로 구현한다:

```python
def process(audio: np.ndarray, sr: int, preset: str = "normal", **kwargs) -> np.ndarray:
    """
    Args:
        audio: 오디오 데이터 (float32, mono)
        sr: 샘플레이트 (16000)
        preset: "light" | "normal" | "strong"
    Returns:
        처리된 오디오 데이터 (float32, mono, 동일 sr)
    """
```

converter.py만 예외: 파일 경로 기반 입출력.

## 모듈별 구현 상세

### 1. converter.py — 오디오 변환/정규화

```python
def convert(input_path: str, output_path: str | None = None) -> tuple[np.ndarray, int]:
    """m4a/mp3/wav → 16kHz mono WAV 변환 + 볼륨 정규화(loudnorm)"""
```

- ffmpeg를 subprocess로 호출
- loudnorm 필터로 볼륨 정규화
- 이미 16kHz mono면 변환 스킵, 디코딩만 수행
- soundfile로 결과 로드하여 ndarray 반환
- ffmpeg 미설치 시: `RuntimeError("ffmpeg not found. Install: brew install ffmpeg")`

### 2. snr.py — SNR 측정/프리셋 자동 선택

```python
def estimate_snr(audio: np.ndarray, sr: int) -> float:
    """WADA SNR 추정 알고리즘으로 SNR(dB) 반환"""

def auto_preset(snr_db: float) -> str:
    """SNR → 프리셋 매핑: >20dB→light, 10~20dB→normal, <10dB→strong"""
```

- WADA SNR: 순수 NumPy/SciPy 구현
- 외부 라이브러리 의존 없음

### 3. dereverb.py — 역반향 제거

```python
def process(audio: np.ndarray, sr: int, preset: str = "normal", **kwargs) -> np.ndarray:
```

- `light` 프리셋: 스킵 (입력 그대로 반환)
- `normal`/`strong`: nara_wpe 적용
- nara_wpe 미설치 시 경고 + 스킵

### 4. denoiser.py — 노이즈 제거

```python
def process(audio: np.ndarray, sr: int, preset: str = "normal", **kwargs) -> np.ndarray:
```

- 기본: DeepFilterNet3 사용
- 폴백: noisereduce (DeepFilterNet 미설치 시)
- 프리셋별 강도:
  - `light`: 경미한 처리
  - `normal`: 표준
  - `strong`: 최대 강도 + highpass 필터(80Hz 이하 제거, 키보드 소음)
- 어떤 엔진을 사용했는지 로그 출력

### 5. agc.py — 동적 볼륨 정규화

```python
def process(audio: np.ndarray, sr: int, preset: str = "normal", **kwargs) -> np.ndarray:
```

- `light` 프리셋: 스킵
- `normal`/`strong`: pyagc 적용
- pyagc 미설치 시 간단한 RMS 기반 정규화로 폴백

## pipeline/__init__.py

파이프라인 전체를 순차 실행하는 헬퍼:

```python
def run_preprocess(input_path: str, preset: str = "normal",
                   on_progress=None) -> tuple[np.ndarray, int]:
    """전처리 파이프라인 전체 실행: convert → snr → dereverb → denoise → agc
    Returns: (audio: np.ndarray, sr: int) — float32, mono, 16kHz"""

def save_preprocessed(audio: np.ndarray, sr: int, output_path: str) -> str:
    """전처리된 ndarray를 임시 WAV 파일로 저장. ASR 모듈이 파일 경로를 요구하므로 브릿지 역할.
    Returns: 저장된 WAV 파일 경로"""
```

on_progress 콜백 시그니처: `(step: str, percent: float, message: str) -> None`

## agc→asr 브릿지

전처리 출력은 ndarray이나 asr.py는 파일 경로를 입력받는다.
`save_preprocessed()`로 임시 WAV를 생성하여 asr에 전달한다.
CLI(transcribe.py)에서의 호출 흐름:

```python
audio, sr = run_preprocess(input_path, preset=args.denoise, on_progress=...)
wav_path = save_preprocessed(audio, sr, tmp_path)
result = transcribe(wav_path, lang=args.lang, context=args.context, on_progress=...)
```

## 프리셋 정리

| 프리셋 | converter | snr | dereverb | denoiser | agc |
|--------|-----------|-----|----------|----------|-----|
| light | 변환+정규화 | 측정 | 스킵 | 경미 | 스킵 |
| normal | 변환+정규화 | 측정 | 적용 | 표준 | 적용 |
| strong | 변환+정규화 | 측정 | 적용 | 최대+highpass | 적용 |
| auto | 변환+정규화 | 측정→프리셋 결정 | 자동 | 자동 | 자동 |
