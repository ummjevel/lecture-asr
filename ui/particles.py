"""파티클 시스템 — 별빛 컬러 파티클을 몬스터볼 주변에 렌더링한다."""

from __future__ import annotations

import random

# 별빛 테마: (문자, 색상) 쌍
_STARS_PROCESSING = [
    ("·", "#aaaaaa"),
    ("·", "#888888"),
    ("✦", "#ffdd66"),
    ("✧", "#ccccdd"),
    ("✧", "#99aacc"),
]

_STARS_ALMOST = [
    ("·", "#cccccc"),
    ("✦", "#ffcc33"),
    ("✦", "#ffee88"),
    ("✧", "#aabbdd"),
    ("✧", "#ddeeff"),
    ("⋆", "#88ccff"),
]

_STARS_COMPLETE = [
    ("✨", "#ffdd00"),
    ("★", "#ffcc00"),
    ("✦", "#ffee66"),
    ("✦", "#ffffff"),
    ("✧", "#88ddff"),
    ("✧", "#aaeeff"),
    ("·", "#ffffcc"),
]


class Particle:
    """개별 파티클."""

    __slots__ = ("x", "y", "char", "color", "vx", "vy", "life")

    def __init__(self, x: int, y: int, char: str, color: str, vx: float, vy: float, life: int):
        self.x = x
        self.y = y
        self.char = char
        self.color = color
        self.vx = vx
        self.vy = vy
        self.life = life


class ParticleSystem:
    """상태별 밀도로 별빛 파티클을 생성하고 매 프레임 갱신한다.

    - processing: 4~6개, 은은한 별빛
    - almost_done: 8~12개, 밝아지는 별빛
    - complete: 15~20개, 화려하게 터짐
    """

    def __init__(self, width: int = 50, height: int = 9):
        self.width = width
        self.height = height
        self.particles: list[Particle] = []
        self._state = "processing"

    # ------------------------------------------------------------------
    def update(self, state: str, progress: float) -> None:
        self._state = state

        if state == "complete":
            target = random.randint(15, 20)
        elif state == "almost_done":
            target = random.randint(8, 12)
        else:
            target = random.randint(4, 6)

        alive: list[Particle] = []
        for p in self.particles:
            p.life -= 1
            if p.life > 0:
                p.x = max(0, min(self.width - 1, int(p.x + p.vx)))
                p.y = max(0, min(self.height - 1, int(p.y + p.vy)))
                alive.append(p)
        self.particles = alive

        while len(self.particles) < target:
            self.particles.append(self._spawn(state))

    # ------------------------------------------------------------------
    def _spawn(self, state: str) -> Particle:
        cx, cy = self.width // 2, self.height // 2

        if state == "complete":
            stars = _STARS_COMPLETE
            x = cx + random.randint(-5, 5)
            y = cy + random.randint(-2, 2)
            vx = random.choice([-3, -2, -1, 1, 2, 3])
            vy = random.choice([-1, 0, 1])
            life = random.randint(4, 8)
        elif state == "almost_done":
            stars = _STARS_ALMOST
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            vx = random.choice([-2, -1, 0, 1, 2])
            vy = random.choice([-1, 0, 1])
            life = random.randint(4, 7)
        else:
            stars = _STARS_PROCESSING
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            vx = random.choice([-1, 0, 0, 1])
            vy = random.choice([0, 0, 0, 1])
            life = random.randint(5, 10)

        char, color = random.choice(stars)
        return Particle(x, y, char, color, vx, vy, life)

    # ------------------------------------------------------------------
    def render(self) -> list[str]:
        """rich markup 포함 그리드 문자열 리스트(height행)."""
        grid: list[list[str]] = [[" "] * self.width for _ in range(self.height)]
        for p in self.particles:
            if 0 <= p.x < self.width and 0 <= p.y < self.height:
                grid[p.y][p.x] = f"[{p.color}]{p.char}[/]"
        return ["".join(row) for row in grid]
