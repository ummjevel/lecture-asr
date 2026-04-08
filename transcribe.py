#!/usr/bin/env python3
"""lecture-asr CLI 진입점.

사용법:
    python transcribe.py lecture.m4a
    python transcribe.py lecture.m4a --format srt --denoise strong
    python transcribe.py ./lectures/ --format both --llm --summary
"""

from __future__ import annotations

# 모델 캐시 디렉토리 설정 — 다른 pipeline import보다 반드시 먼저 실행
from pipeline.cache import setup_cache
setup_cache()

import argparse
import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("lecture-asr")

# 지원 확장자
_AUDIO_EXTS = {".m4a", ".aac", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".opus"}


# ===================================================================
# argparse
# ===================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="강의실 음성 전사 CLI 도구",
    )
    parser.add_argument("input", help="오디오 파일 또는 디렉토리 경로")
    parser.add_argument(
        "--format",
        choices=["txt", "srt", "both"],
        default="txt",
        help="출력 형식 (기본: txt)",
    )
    parser.add_argument(
        "--denoise",
        choices=["light", "normal", "strong", "auto"],
        default="normal",
        help="노이즈 제거 강도 (기본: normal)",
    )
    parser.add_argument(
        "--context",
        help="전공 용어 — 쉼표로 구분 (예: '머신러닝,트랜스포머')",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="LLM 후처리 활성화 (오류 교정, ITN, 단락 분리)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="강의 요약 생성 (--llm 필요)",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Whisper large-v3 교차검증",
    )
    parser.add_argument(
        "--lang",
        default="ko",
        help="언어 코드 (기본: ko)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="디버그 로그 출력",
    )

    args = parser.parse_args(argv)

    # --summary 는 --llm 필요
    if args.summary and not args.llm:
        parser.error("--summary 옵션은 --llm과 함께 사용해야 합니다.")

    return args


# ===================================================================
# 파일 수집
# ===================================================================

def collect_files(path_str: str) -> list[Path]:
    """단일 파일이면 리스트, 디렉토리이면 오디오 파일 목록을 반환한다."""
    p = Path(path_str)
    if p.is_file():
        return [p]
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTS
        )
        if not files:
            logger.error("디렉토리에 오디오 파일이 없습니다: %s", p)
            sys.exit(1)
        return files
    logger.error("파일/디렉토리를 찾을 수 없습니다: %s", p)
    sys.exit(1)


# ===================================================================
# 출력 경로 결정
# ===================================================================

def _output_paths(
    input_path: Path,
    fmt: str,
    summary: bool = False,
) -> list[Path]:
    """출력 파일 경로 목록을 반환한다."""
    stem = input_path.stem
    parent = input_path.parent
    paths: list[Path] = []
    if fmt in ("txt", "both"):
        paths.append(parent / f"{stem}.txt")
    if fmt in ("srt", "both"):
        paths.append(parent / f"{stem}.srt")
    if summary:
        paths.append(parent / f"{stem}.summary.txt")
    return paths


# ===================================================================
# Graceful shutdown
# ===================================================================

_shutdown_requested = False


def _signal_handler(signum: int, frame) -> None:  # noqa: ANN001
    global _shutdown_requested  # noqa: PLW0603
    _shutdown_requested = True
    logger.info("Ctrl+C 감지 — 현재 단계 완료 후 종료합니다...")


# ===================================================================
# 메인 파이프라인
# ===================================================================

def process_file(
    file_path: Path,
    args: argparse.Namespace,
    ui,  # noqa: ANN001 — TranscribeUI
) -> None:
    """단일 오디오 파일의 전사 파이프라인을 실행한다."""
    from pipeline import run_preprocess, save_preprocessed
    from pipeline.asr import transcribe as asr_transcribe, save_result
    from pipeline.postprocess import postprocess
    from pipeline.llm_postprocess import postprocess_with_llm
    from pipeline.cross_validate import cross_validate

    start_time = time.monotonic()
    input_str = str(file_path)

    # 출력 경로 계산
    out_paths = _output_paths(file_path, args.format, summary=args.summary)

    # 임시 WAV 경로
    tmp_dir = tempfile.mkdtemp(prefix="lecture-asr-")
    tmp_wav = os.path.join(tmp_dir, f"{file_path.stem}.wav")

    try:
        # --- 전처리 (1~5단계) ---
        if _shutdown_requested:
            return
        audio, sr = run_preprocess(
            input_str,
            preset=args.denoise,
            on_progress=ui.update_progress,
        )

        if _shutdown_requested:
            return
        wav_path = save_preprocessed(audio, sr, tmp_wav)

        # --- ASR 전사 (6단계) ---
        if _shutdown_requested:
            return
        result = asr_transcribe(
            wav_path,
            lang=args.lang,
            context=args.context,
            on_progress=ui.update_progress,
        )

        # --- 후처리 (7단계) ---
        if _shutdown_requested:
            return
        ui.update_progress("postprocess", 0, "시작")
        if args.llm:
            result = postprocess_with_llm(result, summary=args.summary,
                                          on_progress=ui.update_progress)
        else:
            result = postprocess(result, on_progress=ui.update_progress)
        ui.update_progress("postprocess", 100, "완료")

        # --- 저장 (8단계) ---
        if _shutdown_requested:
            return
        ui.update_progress("save", 0, "저장 중")
        for op in out_paths:
            fmt = "srt" if op.suffix == ".srt" else "txt"
            save_result(result, str(op), format=fmt)
        ui.update_progress("save", 100, "완료")

        # --- 교차검증 (선택) ---
        if args.cross_validate and not _shutdown_requested:
            report = cross_validate(wav_path, result)
            report_path = file_path.parent / f"{file_path.stem}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report if isinstance(report, str) else str(report))
            out_paths.append(report_path)

        elapsed = time.monotonic() - start_time
        ui.stop()
        ui.show_summary(result, elapsed, output_files=[str(p) for p in out_paths])

    finally:
        # 임시 파일 정리
        try:
            if os.path.isfile(tmp_wav):
                os.unlink(tmp_wav)
            os.rmdir(tmp_dir)
        except OSError:
            pass


# ===================================================================
# main
# ===================================================================

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 로깅
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Ctrl+C 핸들러
    signal.signal(signal.SIGINT, _signal_handler)

    # UI 초기화
    from ui.progress import TranscribeUI
    ui = TranscribeUI()

    files = collect_files(args.input)
    total = len(files)

    for idx, file_path in enumerate(files, 1):
        if _shutdown_requested:
            print("\n중단됨 — 이전까지 처리된 결과는 저장되어 있습니다.")
            break

        if total > 1:
            print(f"\n[{idx}/{total}] {file_path.name}")

        ui.show_file_info(
            str(file_path),
            preset=args.denoise,
        )
        ui.start()

        try:
            process_file(file_path, args, ui)
        except KeyboardInterrupt:
            ui.stop()
            print("\n중단됨.")
            break
        except Exception as exc:
            ui.stop()
            logger.error("처리 실패: %s — %s", file_path.name, exc)
            if args.verbose:
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
