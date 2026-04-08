"""메타몽 — 작은 ASCII 아트 + 팔 올림/내림 애니메이션."""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# 프레임 데이터 (5행 — 포켓볼과 높이 맞춤)
# 메타몽: 울퉁불퉁한 윗부분, 점 눈, 얇은 입, 아래로 퍼지는 몸체
# ---------------------------------------------------------------------------

_FRAME_ARMS_DOWN = [
    "          ",
    "  _/~\\_  ",
    " / ·  ·\\ ",
    "(  ‿‿   )",
    " \\___./ ",
]

_FRAME_ARMS_UP = [
    " \\  _ _/ ",
    "  _/~\\_  ",
    " / ·  ·\\ ",
    "(  ‿‿   )",
    " \\___./ ",
]

_FRAME_COMPLETE = [
    "  * _ *  ",
    "  _/~\\_  ",
    " / ^  ^\\ ",
    "(  ▽▽   )",
    " \\___./ ",
]

_ANIM_FRAMES = [_FRAME_ARMS_DOWN, _FRAME_ARMS_UP]

FRAME_INTERVAL = 0.5


class DittoAnimation:
    """메타몽 애니메이션.

    - processing: 팔 올림/내림 반복
    - almost_done: 더 빠르게
    - complete: 기쁜 표정
    """

    def __init__(self) -> None:
        self._frame_idx = 0
        self._last_tick = time.monotonic()
        self._state = "processing"

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    def tick(self) -> None:
        now = time.monotonic()
        interval = FRAME_INTERVAL * (0.5 if self._state == "almost_done" else 1.0)
        if now - self._last_tick >= interval:
            self._frame_idx = (self._frame_idx + 1) % len(_ANIM_FRAMES)
            self._last_tick = now

    def get_frame(self) -> list[str]:
        if self._state == "complete":
            return list(_FRAME_COMPLETE)
        self.tick()
        return list(_ANIM_FRAMES[self._frame_idx])
