"""몬스터볼 픽셀아트 + 4프레임 흔들림 애니메이션."""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# 프레임 데이터 (5행 x 가변폭)
# ---------------------------------------------------------------------------

_FRAME_CENTER = [
    "    🟥🟥🟥    ",
    "  🟥🟥🟥🟥🟥  ",
    "  ⬛⬛🔘⬛⬛  ",
    "  ⬜⬜⬜⬜⬜  ",
    "    ⬜⬜⬜    ",
]

_FRAME_RIGHT = [
    "     🟥🟥🟥   ",
    "   🟥🟥🟥🟥🟥 ",
    "   ⬛⬛◐⬛⬛ ",
    "   ⬜⬜⬜⬜⬜ ",
    "     ⬜⬜⬜   ",
]

_FRAME_LEFT = [
    "   🟥🟥🟥     ",
    " 🟥🟥🟥🟥🟥   ",
    " ⬛⬛◑⬛⬛   ",
    " ⬜⬜⬜⬜⬜   ",
    "   ⬜⬜⬜     ",
]

_FRAME_COMPLETE = [
    "    🟥🟥🟥    ",
    "  🟥🟥🟥🟥🟥  ",
    "  ⬛⬛✨⬛⬛  ",
    "  ⬜⬜⬜⬜⬜  ",
    "    ⬜⬜⬜    ",
]

# 4프레임 루프: 중앙 → 오른쪽 → 중앙 → 왼쪽
_PROCESSING_FRAMES = [_FRAME_CENTER, _FRAME_RIGHT, _FRAME_CENTER, _FRAME_LEFT]

FRAME_INTERVAL = 0.2  # 200ms


class PokeballAnimation:
    """몬스터볼 애니메이션 관리자.

    - state="processing": 4프레임 흔들림 루프 (200ms 간격)
    - state="almost_done": 동일 루프 (더 빠른 간격 가능)
    - state="complete": 🔘→✨ 전환 + "딸깍!" 텍스트
    """

    def __init__(self) -> None:
        self._frame_idx = 0
        self._last_tick = time.monotonic()
        self._state = "processing"
        self._complete_shown = False

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        if value == "complete" and self._state != "complete":
            self._complete_shown = False
        self._state = value

    # ------------------------------------------------------------------
    def tick(self) -> None:
        """시간 경과에 따라 프레임 인덱스를 전진시킨다."""
        now = time.monotonic()
        interval = FRAME_INTERVAL * (0.7 if self._state == "almost_done" else 1.0)
        if now - self._last_tick >= interval:
            self._frame_idx = (self._frame_idx + 1) % len(_PROCESSING_FRAMES)
            self._last_tick = now

    # ------------------------------------------------------------------
    def get_frame(self) -> list[str]:
        """현재 프레임 문자열 리스트(5행)를 반환한다."""
        if self._state == "complete":
            return list(_FRAME_COMPLETE)
        self.tick()
        return list(_PROCESSING_FRAMES[self._frame_idx])

    # ------------------------------------------------------------------
    def get_label(self) -> str:
        """프레임 하단에 표시할 레이블."""
        if self._state == "complete":
            return "딸깍!"
        return ""

    # ------------------------------------------------------------------
    def render(self) -> str:
        """프레임 + 레이블을 합쳐 단일 문자열로 반환한다."""
        lines = self.get_frame()
        label = self.get_label()
        if label:
            # 레이블을 프레임 폭에 맞춰 중앙 정렬
            w = max(len(l) for l in lines) if lines else 0
            lines.append(label.center(w))
        return "\n".join(lines)
