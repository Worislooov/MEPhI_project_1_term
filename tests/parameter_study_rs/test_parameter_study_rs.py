"""Тестовый сценарий: исследование влияния радиуса Шварцшильда rs на кривизну лучей."""

from pathlib import Path
import os
import sys
from typing import List
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
path_to_src = os.path.abspath(os.path.join(current_dir, "../..", "src"))
if path_to_src not in sys.path:
    sys.path.append(path_to_src)

from simulation import PhotonSimulation
from visualizer import Visualizer


def _patched_event_esc(self, phi: float, state: list) -> float:
    """Гладкое отслеживание улета фотона на бесконечность для solve_ivp."""
    u = state[0]
    initial_u = np.sin(self.phi_0) / self.b
    return u - initial_u * 0.5


_patched_event_esc.terminal = True
_patched_event_esc.direction = -1
PhotonSimulation._event_esc = _patched_event_esc


class MirroredPhoton:
    """Зеркальная копия фотона относительно оси X."""

    def __init__(self, sim: PhotonSimulation) -> None:
        self.x = sim.x
        self.y = -sim.y
        self.phi = sim.phi
        self.status = sim.status
        self.b = sim.b
        self.r_s = sim.r_s
        self.r_min = sim.r_min


def simulate_for_rs(rs: float, b_values: List[float]) -> List[object]:
    """Рассчитывает симметричный пучок для конкретного радиуса Шварцшильда."""
    beam = []
    for b in b_values:
        sim = PhotonSimulation(r_s=rs, b=b, phi_0=1e-3, phi_max=6.0 * np.pi)
        try:
            sim.run(rtol=1e-8, atol=1e-10, max_step=0.04)
            beam.append(sim)
            beam.append(MirroredPhoton(sim))
        except (AssertionError, ValueError):
            pass
    return beam


def main() -> None:
    """Сравнительное моделирование трех черных дыр с rs = 0.5, 1.0, 2.0."""
    output_dir = Path(current_dir)
    rs_variants = [0.5, 1.0, 2.0]
    test_b = [float(val) for val in np.linspace(1.0, 8.0, 22)]

    for rs in rs_variants:
        print(f"Моделирование для rs = {rs:.1f}...")
        beam = simulate_for_rs(rs=rs, b_values=test_b)

        vis = Visualizer(fig_size=(16, 9))
        vis.draw_black_hole(rs=rs)
        vis.plot_trajectories(beam, crop_box=16.0)
        vis.set_limits((-14.0, 14.0), (-7.875, 7.875))

        out_file = output_dir / f"study_rs_{rs:.1f}.png"
        vis.save(out_file, dpi=300)
        print(f"Сохранен снимок для rs={rs:.1f}: {out_file.resolve()}")


if __name__ == "__main__":
    main()