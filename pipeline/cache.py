"""모델 캐시 디렉토리 설정.

프로젝트 루트의 .cache/ 폴더에 모든 모델을 저장한다.
다른 모듈보다 먼저 import하여 환경변수를 설정해야 한다.
"""

from __future__ import annotations

import os
from pathlib import Path

# 프로젝트 루트: 이 파일(pipeline/cache.py)의 두 단계 상위
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / ".cache"


def setup_cache() -> Path:
    """모델 캐시 디렉토리를 생성하고 환경변수를 설정한다.

    설정되는 환경변수:
        HF_HOME          — HuggingFace Hub (mlx-qwen3-asr, whisper, punctuation 등)
        TORCH_HOME        — PyTorch 모델 캐시
        XDG_CACHE_HOME    — 기타 라이브러리 범용 캐시

    Returns:
        캐시 디렉토리 경로
    """
    hf_dir = CACHE_DIR / "huggingface"
    torch_dir = CACHE_DIR / "torch"

    hf_dir.mkdir(parents=True, exist_ok=True)
    torch_dir.mkdir(parents=True, exist_ok=True)

    # 이미 사용자가 명시적으로 설정한 경우 덮어쓰지 않음
    os.environ.setdefault("HF_HOME", str(hf_dir))
    os.environ.setdefault("TORCH_HOME", str(torch_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

    return CACHE_DIR
