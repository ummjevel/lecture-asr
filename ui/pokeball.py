"""몬스터볼 — rich 컬러 그라데이션 + 8프레임 역동적 흔들림."""

from __future__ import annotations

import time

# ---------------------------------------------------------------------------
# 색상 팔레트 (순환하며 그라데이션 변화)
# ---------------------------------------------------------------------------

_RED_CYCLE = [
    ['#ff4444', '#cc0000', '#ff4444'],          # 밝-어-밝
    ['#cc0000', '#ff4444', '#cc0000'],          # 어-밝-어
    ['#ff2222', '#ee3333', '#ff2222'],          # 밝2
    ['#dd1111', '#ff4444', '#dd1111'],          # 어2-밝-어2
    ['#ff4444', '#dd1111', '#ff4444'],          # 밝-어2-밝
    ['#cc0000', '#ff2222', '#cc0000'],          # 어-밝2-어
    ['#ee3333', '#cc0000', '#ee3333'],          # 중간
    ['#ff4444', '#990000', '#ff4444'],          # 밝-진-밝
]

_RED5_CYCLE = [
    ['#cc0000', '#ff4444', '#cc0000', '#ff4444', '#990000'],
    ['#ff4444', '#cc0000', '#ff4444', '#990000', '#cc0000'],
    ['#990000', '#ff4444', '#cc0000', '#ff4444', '#cc0000'],
    ['#cc0000', '#990000', '#ff4444', '#cc0000', '#ff4444'],
    ['#ff4444', '#cc0000', '#990000', '#ff4444', '#cc0000'],
    ['#ff2222', '#dd1111', '#ff4444', '#dd1111', '#990000'],
    ['#dd1111', '#ff4444', '#dd1111', '#cc0000', '#ff2222'],
    ['#cc0000', '#ff2222', '#cc0000', '#ff4444', '#dd1111'],
]

_WHITE_CYCLE = [
    ['#e8e8e8', '#d0d0d0', '#e8e8e8', '#d0d0d0', '#e8e8e8'],
    ['#d0d0d0', '#e8e8e8', '#d0d0d0', '#e8e8e8', '#d0d0d0'],
    ['#e0e0e0', '#d8d8d8', '#e8e8e8', '#d8d8d8', '#e0e0e0'],
    ['#d8d8d8', '#e8e8e8', '#d8d8d8', '#e0e0e0', '#d8d8d8'],
    ['#e8e8e8', '#e0e0e0', '#d0d0d0', '#e8e8e8', '#e0e0e0'],
    ['#d0d0d0', '#e0e0e0', '#e8e8e8', '#d0d0d0', '#e8e8e8'],
    ['#e0e0e0', '#e8e8e8', '#d8d8d8', '#e8e8e8', '#d0d0d0'],
    ['#e8e8e8', '#d0d0d0', '#e0e0e0', '#d8d8d8', '#e8e8e8'],
]

_WHITE3_CYCLE = [
    ['#c8c8c8', '#d8d8d8', '#c8c8c8'],
    ['#d8d8d8', '#c8c8c8', '#d8d8d8'],
    ['#c0c0c0', '#d0d0d0', '#c0c0c0'],
    ['#d0d0d0', '#c0c0c0', '#d0d0d0'],
    ['#c8c8c8', '#c0c0c0', '#c8c8c8'],
    ['#c0c0c0', '#c8c8c8', '#d0d0d0'],
    ['#d0d0d0', '#c8c8c8', '#c0c0c0'],
    ['#c8c8c8', '#d0d0d0', '#c8c8c8'],
]

BK = '#222222'
SP = '#ffcc00'
SP2 = '#ffdd44'

# ---------------------------------------------------------------------------
# 프레임 빌더 — 매 프레임 색상이 바뀌는 rich markup 생성
# ---------------------------------------------------------------------------

def _build_row(colors: list[str]) -> str:
    """색상 리스트로 ██ 블록 행 생성."""
    return ''.join(f'[{c}]██[/]' for c in colors)


def _build_frame(color_idx: int, offset: int, tilt: str = 'center') -> list[str]:
    """color_idx에 따라 그라데이션이 바뀌는 포켓볼 프레임."""
    i = color_idx % 8
    pad = ' ' * offset

    r3 = _RED_CYCLE[i]
    r5 = _RED5_CYCLE[i]
    w5 = _WHITE_CYCLE[i]
    w3 = _WHITE3_CYCLE[i]

    # 중앙 버튼 색상도 살짝 변화
    sp = SP if i % 2 == 0 else SP2

    # 기울임에 따른 버튼 모양
    if tilt == 'right':
        btn_char = '◐'
    elif tilt == 'left':
        btn_char = '◑'
    else:
        btn_char = '●'

    mid = f'[{BK}]██[/][{BK}]██[/][{sp}]{btn_char}[/] [{BK}]██[/][{BK}]██[/]'

    return [
        f'{pad}  {_build_row(r3)}',
        f'{pad}{_build_row(r5)}',
        f'{pad}{mid}',
        f'{pad}{_build_row(w5)}',
        f'{pad}  {_build_row(w3)}',
    ]


def _build_complete_frame(color_idx: int) -> list[str]:
    """완료 프레임 — ✨ + 색상 순환."""
    i = color_idx % 8
    pad = ' ' * 12
    r3 = _RED_CYCLE[i]
    r5 = _RED5_CYCLE[i]
    w5 = _WHITE_CYCLE[i]
    w3 = _WHITE3_CYCLE[i]

    mid = f'[{BK}]██[/][{BK}]██[/][#ffdd00]✨[/][{BK}]██[/][{BK}]██[/]'

    return [
        f'{pad}  {_build_row(r3)}',
        f'{pad}{_build_row(r5)}',
        f'{pad}{mid}',
        f'{pad}{_build_row(w5)}',
        f'{pad}  {_build_row(w3)}',
    ]


# ---------------------------------------------------------------------------
# 8프레임 시퀀스: (오프셋, 기울임)
# ---------------------------------------------------------------------------
_SEQUENCE = [
    (12, 'center'),   # 중앙
    (16, 'right'),    # 오른쪽 살짝
    (22, 'right'),    # 오른쪽 크게
    (18, 'right'),    # 되돌아오며
    (12, 'center'),   # 중앙
    (8,  'left'),     # 왼쪽 살짝
    (2,  'left'),     # 왼쪽 크게
    (6,  'left'),     # 되돌아오며
]

FRAME_INTERVAL = 0.35


class PokeballAnimation:
    """몬스터볼 애니메이션 — rich 컬러 그라데이션 + 역동적 흔들림.

    - processing: 8프레임 흔들림 + 색상 순환 (150ms)
    - almost_done: 더 빠르게 (100ms)
    - complete: ✨ + 색상 순환 + "딸깍!"
    """

    def __init__(self) -> None:
        self._frame_idx = 0
        self._color_idx = 0
        self._last_tick = time.monotonic()
        self._state = 'processing'
        self._use_rich = True

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value

    def tick(self) -> None:
        now = time.monotonic()
        interval = FRAME_INTERVAL * (0.65 if self._state == 'almost_done' else 1.0)
        if now - self._last_tick >= interval:
            self._frame_idx = (self._frame_idx + 1) % len(_SEQUENCE)
            self._color_idx += 1
            self._last_tick = now

    def get_frame(self) -> list[str]:
        """rich markup 문자열 리스트(5행)."""
        if self._state == 'complete':
            self._color_idx += 1
            return _build_complete_frame(self._color_idx)
        self.tick()
        offset, tilt = _SEQUENCE[self._frame_idx]
        return _build_frame(self._color_idx, offset, tilt)

    def get_label(self) -> str:
        if self._state == 'complete':
            return '딸깍!'
        return ''

    def render(self) -> str:
        lines = self.get_frame()
        label = self.get_label()
        if label:
            lines.append(f'           {label}')
        return '\n'.join(lines)
