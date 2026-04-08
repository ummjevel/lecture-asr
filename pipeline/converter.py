"""ffmpeg 오디오 변환/정규화 — m4a/mp3/wav → 16kHz mono WAV + loudnorm."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000
_TARGET_CHANNELS = 1


def _ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found. Install: brew install ffmpeg (macOS) "
            "or sudo apt install ffmpeg (Linux)"
        )
    return path


def _probe(input_path: str) -> dict:
    """ffprobe로 오디오 메타데이터 조회."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "a:0",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {}
        import json
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        return streams[0] if streams else {}
    except Exception:
        return {}


def convert(
    input_path: str,
    output_path: str | None = None,
) -> tuple[np.ndarray, int]:
    """m4a/mp3/wav 등을 16kHz mono WAV로 변환하고 loudnorm 볼륨 정규화를 적용한다.

    Args:
        input_path: 입력 오디오 파일 경로.
        output_path: 출력 WAV 경로. None이면 임시 파일 사용.

    Returns:
        (audio, sr) — float32 ndarray와 샘플레이트(16000).
    """
    input_path = str(Path(input_path).resolve())
    ffmpeg = _ffmpeg_bin()

    # 출력 경로 결정
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()
    else:
        output_path = str(Path(output_path).resolve())

    # 소스 정보 확인
    info = _probe(input_path)
    src_sr = int(info.get("sample_rate", 0)) if info else 0
    src_ch = int(info.get("channels", 0)) if info else 0

    logger.info(
        "convert: %s → %s (src %dHz %dch)",
        input_path, output_path, src_sr, src_ch,
    )

    # ffmpeg 명령어 구성
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", str(_TARGET_SR),
        "-ac", str(_TARGET_CHANNELS),
        "-sample_fmt", "s16",
        "-f", "wav",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")

    # soundfile로 로드
    audio, sr = sf.read(output_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    logger.info("convert done: %.1fs, %dHz", len(audio) / sr, sr)
    return audio, sr
