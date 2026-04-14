import time, sys
sys.path.insert(0, '.')
from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.console import Group
from ui.pokeball import PokeballAnimation

anim = PokeballAnimation()
anim.state = 'processing'
c = Console()

with Live(console=c, refresh_per_second=10) as live:
    for _ in range(80):
        lines = anim.get_frame()
        parts = [Text.from_markup(l) for l in lines]
        live.update(Group(*parts))
        time.sleep(0.1)
    anim.state = 'complete'
    for _ in range(20):
        lines = anim.get_frame()
        parts = [Text.from_markup(l) for l in lines]
        label = anim.get_label()
        if label:
            parts.append(Text(f'           {label}', style='bold yellow'))
        live.update(Group(*parts))
        time.sleep(0.1)
