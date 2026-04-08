"""파티클 시스템 — 몬스터볼 주변 15x9 그리드에 파티클을 렌더링한다."""

from __future__ import annotations

import random

PARTICLE_CHARS_PROCESSING = ["·", "✦", "✧"]
PARTICLE_CHARS_ALMOST = ["·", "✦", "✧", "⋆"]
PARTICLE_CHARS_COMPLETE = ["✨", "★", "✦", "✧", "·"]


class Particle:
    """개별 파티클."""

    __slots__ = ("x", "y", "char", "vx", "vy", "life")

    def __init__(self, x: int, y: int, char: str, vx: float, vy: float, life: int):
        self.x = x
        self.y = y
        self.char = char
        self.vx = vx
        self.vy = vy
        self.life = life


class ParticleSystem:
    """상태별 밀도로 파티클을 생성하고 매 프레임 갱신한다.

    - processing: 2~3개, 느리게 떠다님
    - almost_done: 4~5개, 점점 빨라짐
    - complete: 8~10개, 터지듯 퍼짐
    """

    def __init__(self, width: int = 15, height: int = 9):
        self.width = width
        self.height = height
        self.particles: list[Particle] = []
        self._state = "processing"

    # ------------------------------------------------------------------
    def update(self, state: str, progress: float) -> None:
        """파티클 상태 갱신. *state*: processing | almost_done | complete."""
        self._state = state

        # 목표 파티클 수
        if state == "complete":
            target = random.randint(8, 10)
        elif state == "almost_done":
            target = random.randint(4, 5)
        else:
            target = random.randint(2, 3)

        # 기존 파티클 수명 차감 & 이동
        alive: list[Particle] = []
        for p in self.particles:
            p.life -= 1
            if p.life > 0:
                p.x = max(0, min(self.width - 1, int(p.x + p.vx)))
                p.y = max(0, min(self.height - 1, int(p.y + p.vy)))
                alive.append(p)
        self.particles = alive

        # 부족분 채우기
        while len(self.particles) < target:
            self.particles.append(self._spawn(state))

    # ------------------------------------------------------------------
    def _spawn(self, state: str) -> Particle:
        cx, cy = self.width // 2, self.height // 2

        if state == "complete":
            # 중앙 근처에서 바깥으로 터지듯
            chars = PARTICLE_CHARS_COMPLETE
            x = cx + random.randint(-2, 2)
            y = cy + random.randint(-1, 1)
            vx = random.choice([-2, -1, 1, 2])
            vy = random.choice([-1, 0, 1])
            life = random.randint(3, 6)
        elif state == "almost_done":
            chars = PARTICLE_CHARS_ALMOST
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            vx = random.choice([-1, 0, 1])
            vy = random.choice([-1, 0, 1])
            life = random.randint(3, 5)
        else:
            chars = PARTICLE_CHARS_PROCESSING
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            vx = random.choice([-1, 0, 0, 1])
            vy = random.choice([0, 0, 0, 1])
            life = random.randint(4, 8)

        return Particle(x, y, random.choice(chars), vx, vy, life)

    # ------------------------------------------------------------------
    def render(self) -> list[str]:
        """현재 파티클 위치를 반영한 그리드 문자열 리스트(height행)."""
        grid = [[" "] * self.width for _ in range(self.height)]
        for p in self.particles:
            if 0 <= p.x < self.width and 0 <= p.y < self.height:
                grid[p.y][p.x] = p.char
        return ["".join(row) for row in grid]
