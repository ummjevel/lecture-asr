---
name: cli-build
description: "CLI 진입점(transcribe.py)과 터미널 UX(몬스터볼 애니메이션, 파티클, 프로그레스바, 진행상황 출력, 결과 요약)를 구현하는 스킬. rich 라이브러리 기반 터미널 UI 빌드 시 사용."
---

# CLI 진입점 및 UX 모듈 구현 가이드

`transcribe.py`와 `ui/` 디렉토리 하위 모듈을 구현한다.

## 1. transcribe.py — CLI 진입점

### argparse 옵션

```python
parser.add_argument("input", help="오디오 파일 또는 디렉토리 경로")
parser.add_argument("--format", choices=["txt", "srt", "both"], default="txt")
parser.add_argument("--denoise", choices=["light", "normal", "strong", "auto"], default="normal")
parser.add_argument("--context", help="전공 용어 (쉼표 구분)")
parser.add_argument("--llm", action="store_true", help="LLM 후처리 활성화")
parser.add_argument("--summary", action="store_true", help="강의 요약 생성 (--llm 필요)")
parser.add_argument("--cross-validate", action="store_true", help="Whisper 교차검증")
parser.add_argument("--lang", default="ko", help="언어 코드")
```

### 파이프라인 실행 흐름

```python
def main():
    args = parse_args()
    ui = TranscribeUI()  # progress.py

    # 1. 파일 목록 수집 (단일 파일 또는 디렉토리)
    files = collect_files(args.input)

    for file in files:
        ui.show_file_info(file)

        # 2. 전처리
        audio, sr = run_preprocess(file, preset=args.denoise,
                                    on_progress=ui.update_progress)

        # 3. ASR
        result = transcribe(audio_path, lang=args.lang, context=args.context,
                           on_progress=ui.update_progress)

        # 4. 후처리
        if args.llm:
            result = postprocess_with_llm(result, summary=args.summary)
        else:
            result = postprocess(result)

        # 5. 저장
        save_result(result, output_path, format=args.format)

        # 6. 교차검증 (선택)
        if args.cross_validate:
            report = cross_validate(audio_path, result)
            save_report(report, report_path)

        ui.show_summary(result, elapsed)
```

### 진행상황 출력 요구사항
- 각 단계 시작/진행/완료를 실시간으로 표시
- 현재 처리 중인 단계를 시각적으로 하이라이트
- 각 단계의 소요 시간을 측정하여 표시
- 전체 진행률(%)도 함께 표시

## 2. ui/progress.py — 프로그레스 및 진행상황 표시

`rich.live.Live` 기반 실시간 터미널 UI.

### 레이아웃 구조

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
  [6/8] 음성 전사     ━━━━━━━━━━━━━━━━━  67%  8:21 🔴 ← 현재
  [7/8] 후처리        ░░░░░░░░░░░░░░░░░░  --   대기
  [8/8] 저장          ░░░░░░░░░░░░░░░░░░  --   대기

       [몬스터볼 애니메이션 영역]
       [파티클 영역]

  전체: ━━━━━━━━━━━━━━━━━━━━━━━━━━━  67%  9:24 경과
```

### 주요 클래스

```python
class TranscribeUI:
    def __init__(self): ...
    def show_file_info(self, file_path: str): ...
    def update_progress(self, step: str, percent: float, message: str): ...
    def show_summary(self, result, elapsed: float): ...
```

- `update_progress`는 파이프라인 모듈의 on_progress 콜백과 동일 시그니처
- rich.progress.Progress로 각 단계별 바 관리
- 현재 진행 중인 단계 옆에 🔴 또는 spinner 표시
- 완료된 단계는 ✅ + 소요 시간

### 결과 요약 패널

```
╭─ 결과 요약 ──────────────────────────────────────╮
│  ✅ 전사 완료 (12분 34초)                         │
│  📝 lecture.txt  (2,847 단어)                     │
│  🎬 lecture.srt  (342 구간)                       │
│  📋 lecture.summary.txt (요약 생성됨)              │
╰──────────────────────────────────────────────────╯
```

## 3. ui/pokeball.py — 몬스터볼 애니메이션

### 픽셀아트 (5x7 컴팩트)

```python
POKEBALL_FRAMES = [
    # Frame 0: 중앙
    [
        "    🟥🟥🟥    ",
        "  🟥🟥🟥🟥🟥  ",
        "  ⬛⬛🔘⬛⬛  ",
        "  ⬜⬜⬜⬜⬜  ",
        "    ⬜⬜⬜    ",
    ],
    # Frame 1: 오른쪽 기울어짐
    [
        "     🟥🟥🟥   ",
        "   🟥🟥🟥🟥🟥 ",
        "   ⬛⬛🔘⬛⬛ ",
        "   ⬜⬜⬜⬜⬜ ",
        "     ⬜⬜⬜   ",
    ],
    # Frame 2: 중앙 복귀 (Frame 0과 동일)
    # Frame 3: 왼쪽 기울어짐
    [
        "  🟥🟥🟥      ",
        "🟥🟥🟥🟥🟥    ",
        "⬛⬛🔘⬛⬛    ",
        "⬜⬜⬜⬜⬜    ",
        "  ⬜⬜⬜      ",
    ],
]

POKEBALL_COMPLETE = [
    "    🟥🟥🟥    ",
    "  🟥🟥🟥🟥🟥  ",
    "  ⬛⬛✨⬛⬛  ",
    "  ⬜⬜⬜⬜⬜  ",
    "    ⬜⬜⬜    ",
]
```

### 애니메이션 루프

```python
class PokeballAnimation:
    def __init__(self): ...
    def get_frame(self, state: str, progress: float) -> str:
        """state: "processing" | "almost_done" | "complete" """
    def render(self) -> str:
        """현재 프레임 + 파티클을 합쳐서 렌더링"""
```

- 200ms 간격으로 프레임 전환
- state="complete" 시 🔘→✨ 전환 + "딸깍!" 텍스트

## 4. ui/particles.py — 파티클 시스템

### 구현

```python
PARTICLE_CHARS = ["·", "✦", "✧", "✨", "⋆", "★"]

class ParticleSystem:
    def __init__(self, width: int = 15, height: int = 9): ...
    def update(self, state: str, progress: float): ...
    def render(self) -> list[str]:
        """현재 파티클 위치를 반영한 그리드 렌더링"""
```

- 15x9 그리드
- 매 프레임마다 파티클 위치 랜덤 갱신
- 상태별 밀도: processing(2~3개) → almost_done(4~5개, 80%+) → complete(8~10개)
- complete 시 `✨ ★` 위주로 "터지는" 연출

## 5. 폴백

- rich 미설치: 일반 print로 진행상황 표시 (애니메이션 없음)
- 터미널 폭 60 미만: 몬스터볼/파티클 비활성화, 프로그레스바만 표시
- Ctrl+C: 현재까지의 결과를 저장하고 종료
