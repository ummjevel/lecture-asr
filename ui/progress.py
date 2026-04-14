"""rich.live 기반 8단계 프로그레스바 + 결과 요약 패널.

rich 미설치 시 일반 print 폴백을 제공한다.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING

# -------------------------------------------------------------------
# rich 임포트 — 실패 시 _HAS_RICH = False 로 폴백
# -------------------------------------------------------------------
_HAS_RICH = False
try:
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    pass

if TYPE_CHECKING:
    from pipeline.models import TranscriptionResult

# -------------------------------------------------------------------
# 8단계 정의
# -------------------------------------------------------------------

STEPS = [
    ("convert", "오디오 변환"),
    ("snr", "SNR 측정"),
    ("dereverb", "역반향 제거"),
    ("denoise", "노이즈 제거"),
    ("agc", "볼륨 정규화"),
    ("transcribe", "음성 전사"),
    ("postprocess", "후처리"),
    ("save", "저장"),
]

STEP_KEYS = [s[0] for s in STEPS]
STEP_LABELS = {k: v for k, v in STEPS}


def _term_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns


# ===================================================================
# Plain-text 폴백 UI
# ===================================================================

class _PlainUI:
    """rich 없이 print로 진행상황을 표시하는 폴백 UI."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._step_starts: dict[str, float] = {}
        self._completed: set[str] = set()

    def start(self) -> None:
        self._start = time.monotonic()

    def stop(self) -> None:
        pass

    def show_file_info(self, file_path: str, duration: float = 0.0,
                       sample_rate: int = 0, preset: str = "normal",
                       engine: str = "qwen") -> None:
        name = os.path.basename(file_path)
        parts = [f"  파일: {name}"]
        if duration > 0:
            m, s = divmod(int(duration), 60)
            parts.append(f"길이: {m}m {s:02d}s")
        if sample_rate:
            parts.append(f"{sample_rate}Hz")
        parts.append(f"노이즈 제거: {preset}")
        print(" | ".join(parts))

    def update_progress(self, step: str, percent: float, message: str) -> None:
        if step not in self._step_starts:
            self._step_starts[step] = time.monotonic()

        idx = STEP_KEYS.index(step) + 1 if step in STEP_KEYS else "?"
        label = STEP_LABELS.get(step, step)
        bar_len = 20
        filled = int(bar_len * percent / 100)
        bar = "━" * filled + "░" * (bar_len - filled)

        if percent >= 100 and step not in self._completed:
            elapsed = time.monotonic() - self._step_starts[step]
            self._completed.add(step)
            print(f"  [{idx}/8] {label:<10s} {bar} 100% {elapsed:5.1f}s done")
        else:
            print(f"  [{idx}/8] {label:<10s} {bar} {percent:3.0f}% {message}", end="\r")

    def show_summary(self, result: TranscriptionResult, elapsed: float,
                     output_files: list[str] | None = None) -> None:
        m, s = divmod(int(elapsed), 60)
        print(f"\n  전사 완료 ({m}분 {s:02d}초)")
        word_count = len(result.text.split()) if result.text else 0
        seg_count = len(result.segments) if result.segments else 0
        print(f"  {word_count} 단어, {seg_count} 구간")
        if output_files:
            for f in output_files:
                print(f"  -> {f}")


# ===================================================================
# Rich UI
# ===================================================================

_ENGINE_LABELS = {
    "ko-whisper": "Whisper-medium-komixv2 (한국어)",
    "whisper": "Whisper large-v3-turbo",
    "qwen": "Qwen3-ASR-0.6B (bf16)",
}


class _RichUI:
    """rich.live.Live 기반 실시간 터미널 UI."""

    def __init__(self) -> None:
        self._console = Console()
        self._live: Live | None = None
        self._start = time.monotonic()
        self._step_starts: dict[str, float] = {}
        self._step_elapsed: dict[str, float] = {}
        self._step_percent: dict[str, float] = {}
        self._step_message: dict[str, str] = {}
        self._completed: set[str] = set()
        self._current_step: str | None = None
        self._file_info: str = ""
        self._preset: str = "normal"
        self._engine: str = "qwen"

        # 몬스터볼 & 파티클 (터미널 폭 60 이상만)
        self._anim_enabled = _term_width() >= 60
        self._pokeball = None
        self._particles = None
        if self._anim_enabled:
            try:
                from ui.pokeball import PokeballAnimation
                from ui.particles import ParticleSystem
                self._pokeball = PokeballAnimation()
                self._particles = ParticleSystem()
            except Exception:
                self._anim_enabled = False

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._start = time.monotonic()
        # 시작 직후부터 애니메이션 표시
        if self._pokeball:
            self._pokeball.state = "processing"
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
            auto_refresh=False,  # 수동 + 타이머 기반 갱신
        )
        self._live.start()
        # 별도 스레드에서 100ms마다 렌더링 갱신 (ffmpeg 등 블로킹 중에도 애니메이션 동작)
        import threading
        self._refresh_stop = threading.Event()
        self._refresh_thread = threading.Thread(target=self._auto_refresh_loop, daemon=True)
        self._refresh_thread.start()

    def _auto_refresh_loop(self) -> None:
        """백그라운드 스레드에서 주기적으로 UI를 갱신한다."""
        while not self._refresh_stop.is_set():
            try:
                # 파티클 상태 갱신 (매 프레임)
                if self._particles:
                    overall = self._overall_percent()
                    state = "complete" if overall >= 100 else ("almost_done" if overall >= 80 else "processing")
                    self._particles.update(state, overall)
                if self._live:
                    self._live.update(self._render())
                    self._live.refresh()
            except Exception:
                pass
            self._refresh_stop.wait(0.1)

    def stop(self) -> None:
        # 1) 먼저 Live를 멈춰서 백그라운드 스레드가 더 이상 self._live.refresh()를
        #    호출하지 못하게 한다. 백그라운드 루프는 예외를 swallow하므로 안전.
        live = self._live
        self._live = None
        if hasattr(self, '_refresh_stop'):
            self._refresh_stop.set()
            self._refresh_thread.join(timeout=2)
            if self._refresh_thread.is_alive():
                # 좀비 스레드 — daemon이므로 프로세스 종료 시 회수되지만 로깅
                import logging
                logging.getLogger(__name__).warning(
                    "UI refresh thread did not exit within timeout"
                )
        if live:
            try:
                live.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def show_file_info(self, file_path: str, duration: float = 0.0,
                       sample_rate: int = 0, preset: str = "normal",
                       engine: str = "qwen") -> None:
        name = os.path.basename(file_path)
        parts: list[str] = []
        if duration > 0:
            m, s = divmod(int(duration), 60)
            parts.append(f"{m}m {s:02d}s")
        if sample_rate:
            parts.append(f"{sample_rate}Hz, mono")
        meta = ", ".join(parts)
        self._file_info = f"[bold]{name}[/bold]" + (f" ({meta})" if meta else "")
        self._preset = preset
        self._engine = engine
        self._refresh()

    # ------------------------------------------------------------------
    def update_progress(self, step: str, percent: float, message: str) -> None:
        now = time.monotonic()
        if step not in self._step_starts:
            self._step_starts[step] = now

        self._step_percent[step] = percent
        self._step_message[step] = message
        self._current_step = step

        if percent >= 100 and step not in self._completed:
            self._step_elapsed[step] = now - self._step_starts[step]
            self._completed.add(step)

        # 애니메이션 상태 갱신
        overall = self._overall_percent()
        anim_state = "complete" if overall >= 100 else ("almost_done" if overall >= 80 else "processing")
        if self._pokeball:
            self._pokeball.state = anim_state
        if self._particles:
            self._particles.update(anim_state, overall)

        self._refresh()

    # ------------------------------------------------------------------
    def _overall_percent(self) -> float:
        if not self._step_percent:
            return 0.0
        total = sum(self._step_percent.get(k, 0.0) for k in STEP_KEYS)
        return total / len(STEP_KEYS)

    # ------------------------------------------------------------------
    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    # ------------------------------------------------------------------
    def _render(self) -> Panel:
        """전체 레이아웃을 렌더링한다."""
        parts: list = []

        # --- 파일 정보 헤더 ---
        if self._file_info:
            header_lines = [
                Text.from_markup(f"  \U0001f4c1 {self._file_info}"),
                Text.from_markup(f"  \U0001f527 노이즈 제거: {self._preset}"),
                Text.from_markup(f"  \U0001f9e0 모델: {_ENGINE_LABELS.get(self._engine, self._engine)}"),
            ]
            for line in header_lines:
                parts.append(line)
            parts.append(Text(""))

        # --- 8단계 프로그레스 ---
        for i, (key, label) in enumerate(STEPS, 1):
            pct = self._step_percent.get(key, 0.0)
            done = key in self._completed

            # 바 렌더링
            bar_w = 18
            filled = int(bar_w * pct / 100)
            bar_full = "━" * filled
            bar_empty = "░" * (bar_w - filled)

            if done:
                elapsed = self._step_elapsed.get(key, 0)
                m, s = divmod(int(elapsed), 60)
                time_str = f"{m}:{s:02d}" if m else f"0:{s:02d}"
                status = "\u2705"
                line = f"  [dim][{i}/8][/dim] {label:<8s} [green]{bar_full}[/green]{bar_empty} [green]100%[/green] {time_str:>5s} {status}"
            elif key == self._current_step:
                elapsed = time.monotonic() - self._step_starts.get(key, time.monotonic())
                m, s = divmod(int(elapsed), 60)
                time_str = f"{m}:{s:02d}" if m else f"0:{s:02d}"
                status = "\U0001f534"
                line = f"  [bold][{i}/8][/bold] {label:<8s} [cyan]{bar_full}[/cyan]{bar_empty} [cyan]{pct:3.0f}%[/cyan] {time_str:>5s} {status}"
            else:
                line = f"  [dim][{i}/8] {label:<8s} {bar_empty}  --   대기[/dim]"

            parts.append(Text.from_markup(line))

        # --- 몬스터볼 + 파티클 ---
        if self._anim_enabled and self._pokeball:
            parts.append(Text(""))
            poke_lines = self._pokeball.get_frame()
            label = self._pokeball.get_label()

            if self._particles:
                particle_rows = self._particles.render()
                merged = self._merge_pokeball_particles(poke_lines, particle_rows)
                for ml in merged:
                    parts.append(Text.from_markup(f"    {ml}"))
            else:
                for pl in poke_lines:
                    parts.append(Text.from_markup(f"    {pl}"))

            if label:
                parts.append(Text(f"           {label}", style="bold yellow"))

        # --- 전체 진행률 ---
        parts.append(Text(""))
        overall = self._overall_percent()
        elapsed_total = time.monotonic() - self._start
        m, s = divmod(int(elapsed_total), 60)
        bar_w = 27
        filled = int(bar_w * overall / 100)
        bar_full = "━" * filled
        bar_empty = "░" * (bar_w - filled)
        parts.append(Text.from_markup(
            f"  전체: [cyan]{bar_full}[/cyan]{bar_empty} {overall:3.0f}%  {m}:{s:02d} 경과"
        ))

        return Panel(
            Group(*parts),
            title="강의 녹음 전사",
            border_style="blue",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_pokeball_particles(
        poke_lines: list[str], particle_rows: list[str]
    ) -> list[str]:
        """몬스터볼을 파티클 그리드 중앙에 합성한다."""
        ph = len(poke_lines)  # 5
        gh = len(particle_rows)  # 9
        gw = len(particle_rows[0]) if particle_rows else 15

        # 몬스터볼 시작 y (그리드 중앙)
        y_off = max(0, (gh - ph) // 2)

        merged: list[str] = []
        for y in range(gh):
            row = list(particle_rows[y]) if y < len(particle_rows) else [" "] * gw
            if y_off <= y < y_off + ph:
                poke_row = poke_lines[y - y_off]
                merged.append(poke_row)
            else:
                merged.append("".join(row))
        return merged

    # ------------------------------------------------------------------
    def show_summary(self, result: TranscriptionResult, elapsed: float,
                     output_files: list[str] | None = None) -> None:
        """결과 요약 패널을 출력한다."""
        m, s = divmod(int(elapsed), 60)

        lines: list[str] = [f"  \u2705 전사 완료 ({m}분 {s:02d}초)"]

        word_count = len(result.text.split()) if result.text else 0
        seg_count = len(result.segments) if result.segments else 0

        if output_files:
            for f in output_files:
                base = os.path.basename(f)
                if f.endswith(".txt") and not f.endswith(".summary.txt"):
                    lines.append(f"  \U0001f4dd {base}  ({word_count:,} 단어)")
                elif f.endswith(".srt"):
                    lines.append(f"  \U0001f3ac {base}  ({seg_count} 구간)")
                elif f.endswith(".summary.txt"):
                    lines.append(f"  \U0001f4cb {base} (요약 생성됨)")
                else:
                    lines.append(f"  \U0001f4c4 {base}")
        else:
            lines.append(f"  \U0001f4dd {word_count:,} 단어, {seg_count} 구간")

        panel = Panel(
            "\n".join(lines),
            title="결과 요약",
            border_style="green",
        )
        self._console.print(panel)


# ===================================================================
# 배치 처리 UI
# ===================================================================

# 파일 상태 상수
_ST_PENDING = "pending"
_ST_PREPROCESS = "preprocess"
_ST_ASR = "asr"
_ST_POSTPROCESS = "postprocess"
_ST_DONE = "done"
_ST_ERROR = "error"

_STATUS_ICONS = {
    _ST_PENDING: "\u23f3",      # ⏳
    _ST_PREPROCESS: "\U0001f527",  # 🔧
    _ST_ASR: "\U0001f9e0",      # 🧠
    _ST_POSTPROCESS: "\u270f\ufe0f",  # ✏️
    _ST_DONE: "\u2705",         # ✅
    _ST_ERROR: "\u274c",        # ❌
}

_STATUS_LABELS = {
    _ST_PENDING: "대기",
    _ST_PREPROCESS: "전처리 중",
    _ST_ASR: "전사 중",
    _ST_POSTPROCESS: "후처리 중",
    _ST_DONE: "완료",
    _ST_ERROR: "실패",
}


class _BatchPlainUI:
    """rich 없이 배치 진행상황을 표시하는 폴백 UI."""

    def __init__(self, file_names: list[str], workers: int) -> None:
        self._files = file_names
        self._workers = workers
        self._statuses: dict[str, str] = {f: _ST_PENDING for f in file_names}
        self._messages: dict[str, str] = {}
        self._start = time.monotonic()

    def start(self) -> None:
        self._start = time.monotonic()
        n = len(self._files)
        print(f"\n  배치 모드: {n}개 파일, {self._workers} 워커")

    def stop(self) -> None:
        pass

    def update_file(self, file_name: str, status: str, message: str = "") -> None:
        self._statuses[file_name] = status
        self._messages[file_name] = message
        icon = _STATUS_ICONS.get(status, "?")
        label = _STATUS_LABELS.get(status, status)
        print(f"  {icon} {file_name}: {label}" + (f" — {message}" if message else ""))

    def show_batch_summary(
        self,
        results: dict[str, object],
        elapsed: float,
        output_files: dict[str, list[str]],
    ) -> None:
        m, s = divmod(int(elapsed), 60)
        done = sum(1 for st in self._statuses.values() if st == _ST_DONE)
        total = len(self._files)
        print(f"\n  배치 완료: {done}/{total} 성공 ({m}분 {s:02d}초)")
        for fname, paths in output_files.items():
            for p in paths:
                print(f"  -> {p}")


class _BatchRichUI:
    """rich.live 기반 배치 진행상황 UI."""

    def __init__(self, file_names: list[str], workers: int) -> None:
        self._console = Console()
        self._files = file_names
        self._workers = workers
        self._statuses: dict[str, str] = {f: _ST_PENDING for f in file_names}
        self._messages: dict[str, str] = {f: "" for f in file_names}
        self._start = time.monotonic()
        self._live: Live | None = None
        self._phase: str = ""  # "preprocess" | "asr" | ""
        self._phase_progress: float = 0.0  # 현재 phase 내 진행률

    def start(self) -> None:
        self._start = time.monotonic()
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def set_phase(self, phase: str, progress: float = 0.0) -> None:
        self._phase = phase
        self._phase_progress = progress
        self._refresh()

    def update_file(self, file_name: str, status: str, message: str = "") -> None:
        self._statuses[file_name] = status
        self._messages[file_name] = message
        self._refresh()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Panel:
        parts: list = []
        total = len(self._files)
        done = sum(1 for st in self._statuses.values() if st == _ST_DONE)
        errors = sum(1 for st in self._statuses.values() if st == _ST_ERROR)

        # 헤더
        parts.append(Text.from_markup(
            f"  [bold]{total}[/bold]개 파일  |  "
            f"[bold]{self._workers}[/bold] 워커  |  "
            f"Phase: [cyan]{self._phase or '준비'}[/cyan]"
        ))
        parts.append(Text(""))

        # 파일별 상태
        for fname in self._files:
            st = self._statuses[fname]
            msg = self._messages.get(fname, "")
            icon = _STATUS_ICONS.get(st, "?")
            label = _STATUS_LABELS.get(st, st)

            # 파일명 자르기 (너무 길면)
            display_name = fname if len(fname) <= 30 else f"...{fname[-27:]}"

            if st == _ST_DONE:
                line = f"  {icon} [green]{display_name:<30s}[/green] [green]{label}[/green]"
            elif st == _ST_ERROR:
                line = f"  {icon} [red]{display_name:<30s}[/red] [red]{label}[/red]"
                if msg:
                    line += f" [dim]({msg})[/dim]"
            elif st == _ST_PENDING:
                line = f"  {icon} [dim]{display_name:<30s} {label}[/dim]"
            else:
                line = f"  {icon} [bold]{display_name:<30s}[/bold] [cyan]{label}[/cyan]"
                if msg:
                    line += f" [dim]— {msg}[/dim]"

            parts.append(Text.from_markup(line))

        # 전체 진행률 바
        parts.append(Text(""))
        elapsed = time.monotonic() - self._start
        m, s = divmod(int(elapsed), 60)
        overall = (done / total * 100) if total > 0 else 0
        bar_w = 27
        filled = int(bar_w * overall / 100)
        bar_full = "━" * filled
        bar_empty = "░" * (bar_w - filled)
        parts.append(Text.from_markup(
            f"  전체: [cyan]{bar_full}[/cyan]{bar_empty} "
            f"{done}/{total} 완료"
            + (f"  [red]{errors} 실패[/red]" if errors else "")
            + f"  {m}:{s:02d} 경과"
        ))

        return Panel(
            Group(*parts),
            title="배치 전사",
            border_style="blue",
        )

    def show_batch_summary(
        self,
        results: dict[str, object],
        elapsed: float,
        output_files: dict[str, list[str]],
    ) -> None:
        m, s = divmod(int(elapsed), 60)
        done = sum(1 for st in self._statuses.values() if st == _ST_DONE)
        errors = sum(1 for st in self._statuses.values() if st == _ST_ERROR)
        total = len(self._files)

        lines: list[str] = [
            f"  \u2705 배치 완료: {done}/{total} 성공 ({m}분 {s:02d}초)"
        ]
        if errors:
            lines.append(f"  \u274c {errors}개 실패")

        for fname in self._files:
            paths = output_files.get(fname, [])
            if paths:
                lines.append(f"  \U0001f4c1 {fname}")
                for p in paths:
                    base = os.path.basename(p)
                    lines.append(f"     -> {base}")

        panel = Panel(
            "\n".join(lines),
            title="배치 결과 요약",
            border_style="green",
        )
        self._console.print(panel)


class BatchUI:
    """배치 처리 UI. rich 유무에 따라 자동 선택."""

    def __init__(self, file_names: list[str], workers: int) -> None:
        if _HAS_RICH:
            self._impl = _BatchRichUI(file_names, workers)
        else:
            self._impl = _BatchPlainUI(file_names, workers)

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def set_phase(self, phase: str, progress: float = 0.0) -> None:
        if hasattr(self._impl, "set_phase"):
            self._impl.set_phase(phase, progress)

    def update_file(self, file_name: str, status: str, message: str = "") -> None:
        self._impl.update_file(file_name, status, message)

    def show_batch_summary(
        self,
        results: dict[str, object],
        elapsed: float,
        output_files: dict[str, list[str]],
    ) -> None:
        self._impl.show_batch_summary(results, elapsed, output_files)


# ===================================================================
# 공개 클래스 — rich 유무에 따라 자동 선택
# ===================================================================

class TranscribeUI:
    """파이프라인 진행상황 UI. rich가 있으면 Live UI, 없으면 plain print."""

    def __init__(self) -> None:
        if _HAS_RICH:
            self._impl = _RichUI()
        else:
            self._impl = _PlainUI()

    def start(self) -> None:
        self._impl.start()

    def stop(self) -> None:
        self._impl.stop()

    def show_file_info(self, file_path: str, duration: float = 0.0,
                       sample_rate: int = 0, preset: str = "normal",
                       engine: str = "qwen") -> None:
        self._impl.show_file_info(file_path, duration=duration,
                                  sample_rate=sample_rate, preset=preset,
                                  engine=engine)

    def update_progress(self, step: str, percent: float, message: str) -> None:
        self._impl.update_progress(step, percent, message)

    def show_summary(self, result: TranscriptionResult, elapsed: float,
                     output_files: list[str] | None = None) -> None:
        self._impl.show_summary(result, elapsed, output_files=output_files)
