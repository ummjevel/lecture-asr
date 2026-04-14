#!/usr/bin/env python3
"""lecture-asr CLI 진입점.

사용법:
    python transcribe.py lecture.m4a
    python transcribe.py lecture.m4a --format srt --denoise strong
    python transcribe.py ./lectures/ --format both --llm --summary
"""

from __future__ import annotations

# 환경변수 — 가장 먼저 설정
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # OpenMP 중복 로딩 우회 (mlx + torch)

# 모델 캐시 디렉토리 설정 — 다른 pipeline import보다 반드시 먼저 실행
from pipeline.cache import setup_cache
setup_cache()

import argparse
import logging
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
        "--lightweight",
        action="store_true",
        help="경량 노이즈 제거 (noisereduce만 사용, 메모리 절약)",
    )
    parser.add_argument(
        "--engine",
        choices=["qwen", "whisper", "ko-whisper"],
        default="qwen",
        help="ASR 엔진 (기본: qwen, whisper: Whisper large-v3, ko-whisper: 한국어 특화 Whisper)",
    )
    parser.add_argument(
        "--asr-only",
        action="store_true",
        help="전처리 스킵, ASR만 실행 (이미 WAV인 경우 테스트용)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="배치 병렬 전처리 워커 수 (기본: auto, 0=자동)",
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
_ctrl_c_count = 0


def _signal_handler(signum: int, frame) -> None:  # noqa: ANN001
    global _shutdown_requested, _ctrl_c_count  # noqa: PLW0603
    _ctrl_c_count += 1
    _shutdown_requested = True
    if _ctrl_c_count >= 2:
        print("\n강제 종료합니다.")
        sys.exit(1)
    print("\nCtrl+C 감지 — 한 번 더 누르면 즉시 종료합니다.")


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
    from pipeline.asr import transcribe as asr_transcribe, transcribe_whisper, transcribe_ko_whisper, save_result
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
        if args.asr_only:
            # 전처리 스킵 — WAV면 그대로, 아니면 변환만
            if file_path.suffix.lower() == ".wav":
                for step in ["convert", "snr", "dereverb", "denoise", "agc"]:
                    ui.update_progress(step, 100.0, "스킵 (--asr-only)")
                wav_path = input_str
            else:
                from pipeline.converter import convert
                ui.update_progress("convert", 5.0, "WAV 변환 중...")
                audio, sr = convert(input_str)
                ui.update_progress("convert", 100.0, f"변환 완료: {len(audio) / sr:.1f}초")
                for step in ["snr", "dereverb", "denoise", "agc"]:
                    ui.update_progress(step, 100.0, "스킵 (--asr-only)")
                wav_path = save_preprocessed(audio, sr, tmp_wav)
        else:
            # --- 전처리 (1~5단계) ---
            if _shutdown_requested:
                return
            audio, sr = run_preprocess(
                input_str,
                preset=args.denoise,
                denoise_engine="lightweight" if args.lightweight else "auto",
                on_progress=ui.update_progress,
            )

            if _shutdown_requested:
                return
            wav_path = save_preprocessed(audio, sr, tmp_wav)

        # --- ASR 전사 (6단계) ---
        if _shutdown_requested:
            return
        if args.engine == "ko-whisper":
            result = transcribe_ko_whisper(
                wav_path,
                lang=args.lang,
                on_progress=ui.update_progress,
            )
        elif args.engine == "whisper":
            result = transcribe_whisper(
                wav_path,
                lang=args.lang,
                on_progress=ui.update_progress,
            )
        else:
            result = asr_transcribe(
                wav_path,
                lang=args.lang,
                context=args.context,
                on_progress=ui.update_progress,
            )

        # --- 후처리 (7단계) ---
        if _shutdown_requested:
            return
        if args.engine in ("whisper", "ko-whisper"):
            # Whisper 계열은 이미 깨끗한 텍스트 — hallucination 제거 + 필러 제거만
            ui.update_progress("postprocess", 50.0, "후처리 중...")
            from pipeline.postprocess import _remove_fillers, _remove_hallucination, _normalize_whitespace
            import copy
            processed = copy.deepcopy(result)
            processed.text = _normalize_whitespace(_remove_hallucination(_remove_fillers(processed.text)))
            for seg in processed.segments:
                seg.text = _normalize_whitespace(_remove_hallucination(_remove_fillers(seg.text)))
            processed.segments = [s for s in processed.segments if s.text.strip()]
            result = processed
            ui.update_progress("postprocess", 100.0, "완료")
        else:
            ui.update_progress("postprocess", 5.0, "후처리 모델 로딩 중...")
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
        ui.update_progress("save", 100.0, "완료!")
        # 완료 애니메이션 (✨ + 딸깍!) 표시 후 종료
        time.sleep(3)
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
# 배치 전처리 워커 (별도 프로세스에서 실행)
# ===================================================================

def _preprocess_worker(
    input_path: str,
    output_wav: str,
    preset: str,
    denoise_engine: str,
) -> tuple[str, str | None]:
    """전처리를 실행하고 결과 WAV 경로를 반환한다.

    Returns:
        (file_name, wav_path) — 성공 시 wav_path, 실패 시 None.
    """
    file_name = os.path.basename(input_path)
    try:
        from pipeline import run_preprocess, save_preprocessed
        audio, sr = run_preprocess(input_path, preset=preset, denoise_engine=denoise_engine)
        wav_path = save_preprocessed(audio, sr, output_wav)
        return (file_name, wav_path)
    except Exception as exc:
        logger.error("전처리 실패: %s — %s", file_name, exc)
        return (file_name, None)


def _resolve_workers(requested: int, file_count: int) -> int:
    """워커 수를 결정한다."""
    if requested > 0:
        return min(requested, file_count)
    cpu = os.cpu_count() or 4
    return min(cpu // 2, file_count, 3)


# ===================================================================
# 배치 파이프라인
# ===================================================================

def process_batch(
    files: list[Path],
    args: argparse.Namespace,
) -> None:
    """여러 파일을 Phase1(전처리 병렬) → Phase2(ASR 순차) → 저장으로 처리한다."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from pipeline.asr import transcribe as asr_transcribe, transcribe_whisper, transcribe_ko_whisper, save_result
    from pipeline.postprocess import postprocess
    from pipeline.llm_postprocess import postprocess_with_llm
    from pipeline.cross_validate import cross_validate
    from ui.progress import BatchUI

    file_names = [f.name for f in files]
    workers = _resolve_workers(args.workers, len(files))

    batch_ui = BatchUI(file_names, workers)
    batch_ui.start()
    batch_start = time.monotonic()

    # 임시 디렉토리
    tmp_dir = tempfile.mkdtemp(prefix="lecture-asr-batch-")

    # wav_path 매핑: file_name → wav_path
    wav_map: dict[str, str] = {}
    results: dict[str, object] = {}
    output_files: dict[str, list[str]] = {}

    try:
        # ============================================================
        # Phase 1: 전처리 병렬
        # ============================================================
        batch_ui.set_phase("전처리 (병렬)")

        if args.asr_only:
            # 전처리 스킵
            for fp in files:
                fname = fp.name
                if fp.suffix.lower() == ".wav":
                    wav_map[fname] = str(fp)
                else:
                    from pipeline.converter import convert
                    from pipeline import save_preprocessed
                    batch_ui.update_file(fname, "preprocess", "WAV 변환 중...")
                    audio, sr = convert(str(fp))
                    tmp_wav = os.path.join(tmp_dir, f"{fp.stem}.wav")
                    wav_map[fname] = save_preprocessed(audio, sr, tmp_wav)
                batch_ui.update_file(fname, "done" if args.asr_only else "preprocess", "전처리 스킵")
        else:
            # 병렬 전처리
            futures = {}
            executor = ProcessPoolExecutor(max_workers=workers)
            try:
                for fp in files:
                    fname = fp.name
                    tmp_wav = os.path.join(tmp_dir, f"{fp.stem}.wav")
                    batch_ui.update_file(fname, "preprocess", "대기 중...")
                    future = executor.submit(
                        _preprocess_worker,
                        str(fp),
                        tmp_wav,
                        args.denoise,
                        "lightweight" if args.lightweight else "auto",
                    )
                    futures[future] = fname

                for future in as_completed(futures):
                    if _shutdown_requested:
                        for f in futures:
                            f.cancel()
                        break
                    fname = futures[future]
                    try:
                        file_name, wav_path = future.result()
                    except Exception as exc:
                        batch_ui.update_file(fname, "error", f"전처리 실패: {exc}")
                        continue
                    if wav_path:
                        wav_map[fname] = wav_path
                        batch_ui.update_file(fname, "preprocess", "전처리 완료")
                    else:
                        batch_ui.update_file(fname, "error", "전처리 실패")
            finally:
                # 셧다운 요청 시 워커 즉시 종료, 정상 시 완료 대기
                executor.shutdown(
                    wait=not _shutdown_requested,
                    cancel_futures=_shutdown_requested,
                )

        if _shutdown_requested:
            batch_ui.stop()
            print("\n중단됨.")
            return

        # ============================================================
        # Phase 2: ASR 순차
        # ============================================================
        batch_ui.set_phase("ASR 전사 (순차)")

        for fp in files:
            if _shutdown_requested:
                break
            fname = fp.name
            wav_path = wav_map.get(fname)
            if not wav_path:
                continue  # 전처리 실패한 파일은 스킵

            batch_ui.update_file(fname, "asr", "전사 중...")

            try:
                if args.engine == "ko-whisper":
                    result = transcribe_ko_whisper(wav_path, lang=args.lang)
                elif args.engine == "whisper":
                    result = transcribe_whisper(wav_path, lang=args.lang)
                else:
                    result = asr_transcribe(
                        wav_path, lang=args.lang, context=args.context,
                    )
                results[fname] = result
                batch_ui.update_file(fname, "postprocess", "후처리 중...")
            except Exception as exc:
                batch_ui.update_file(fname, "error", str(exc)[:50])
                logger.error("ASR 실패: %s — %s", fname, exc)
                continue

            # --- 후처리 ---
            try:
                if args.engine in ("whisper", "ko-whisper"):
                    from pipeline.postprocess import _remove_fillers, _remove_hallucination, _normalize_whitespace
                    import copy
                    processed = copy.deepcopy(result)
                    processed.text = _normalize_whitespace(_remove_hallucination(_remove_fillers(processed.text)))
                    for seg in processed.segments:
                        seg.text = _normalize_whitespace(_remove_hallucination(_remove_fillers(seg.text)))
                    processed.segments = [s for s in processed.segments if s.text.strip()]
                    result = processed
                elif args.llm:
                    result = postprocess_with_llm(result, summary=args.summary)
                else:
                    result = postprocess(result)

                results[fname] = result
            except Exception as exc:
                logger.error("후처리 실패: %s — %s", fname, exc)

            # --- 저장 ---
            out_paths = _output_paths(fp, args.format, summary=args.summary)
            saved: list[str] = []
            for op in out_paths:
                fmt = "srt" if op.suffix == ".srt" else "txt"
                save_result(result, str(op), format=fmt)
                saved.append(str(op))

            # --- 교차검증 ---
            if args.cross_validate:
                try:
                    report = cross_validate(wav_path, result)
                    report_path = fp.parent / f"{fp.stem}_report.txt"
                    with open(report_path, "w", encoding="utf-8") as f:
                        f.write(report if isinstance(report, str) else str(report))
                    saved.append(str(report_path))
                except Exception as exc:
                    logger.error("교차검증 실패: %s — %s", fname, exc)

            output_files[fname] = saved
            batch_ui.update_file(fname, "done", "완료")

        # ============================================================
        # 결과 요약
        # ============================================================
        elapsed = time.monotonic() - batch_start
        batch_ui.stop()
        batch_ui.show_batch_summary(results, elapsed, output_files)

    finally:
        # 임시 파일 정리
        try:
            for wav in wav_map.values():
                if wav.startswith(tmp_dir) and os.path.isfile(wav):
                    os.unlink(wav)
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

    files = collect_files(args.input)
    total = len(files)

    # 배치 모드: 파일 2개 이상
    if total >= 2:
        process_batch(files, args)
        return

    # 단일 파일 모드 (기존 동작)
    from ui.progress import TranscribeUI
    ui = TranscribeUI()

    for idx, file_path in enumerate(files, 1):
        if _shutdown_requested:
            print("\n중단됨 — 이전까지 처리된 결과는 저장되어 있습니다.")
            break

        ui.show_file_info(
            str(file_path),
            preset=args.denoise,
            engine=args.engine,
        )
        ui.start()

        try:
            process_file(file_path, args, ui)
        except KeyboardInterrupt:
            print("\n중단됨.")
            break
        except Exception as exc:
            logger.error("처리 실패: %s — %s", file_path.name, exc)
            if args.verbose:
                import traceback
                traceback.print_exc()
        finally:
            ui.stop()  # 항상 백그라운드 스레드/Live 해제 — 멱등 호출


if __name__ == "__main__":
    main()
