"""Тестовый сценарий: гравитационное линзирование и каустика за черной дырой."""

from pathlib import Path
import time
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


def run_lensing_beam(rs: float) -> List[PhotonSimulation]:
    """Генерирует пучок с акцентом на пролетные отклоняющиеся траектории."""
    b_values = np.concatenate([
        [1.2, 2.0],
        np.linspace(2.64, 3.20, 16),
        np.linspace(3.30, 5.00, 12),
        np.linspace(5.20, 7.50, 8)
    ])
    b_sorted = sorted([float(b) for b in b_values])

    completed = []
    print(f"Расчет гравитационного линзирования для {len(b_sorted)} лучей...")
    for b in b_sorted:
        sim = PhotonSimulation(r_s=rs, b=b, phi_0=1e-3, phi_max=8.0 * np.pi)
        try:
            sim.run(rtol=1e-8, atol=1e-10, max_step=0.035)
            completed.append(sim)
            print(f"b = {b:5.2f} rs | {sim.status:<22} | r_min = {sim.r_min:.3f} rs")
        except (AssertionError, ValueError) as err:
            print(f"b = {b:5.2f} rs | Ошибка ({err}) — пропуск")

    return completed


def main() -> None:
    """Визуализация перекрестия лучей и каустики за горизонтом событий."""
    rs_val = 1.0
    output_dir = Path(current_dir)

    sims = run_lensing_beam(rs=rs_val)
    full_beam = []
    for sim in sims:
        full_beam.append(sim)
        full_beam.append(MirroredPhoton(sim))

    vis = Visualizer(fig_size=(16, 9))
    vis.draw_black_hole(rs=rs_val)
    vis.plot_trajectories(full_beam, crop_box=18.0)
    vis.set_limits((-8.0, 16.0), (-6.75, 6.75))

    img_path = output_dir / "gravitational_lensing.png"
    vis.save(img_path, dpi=300)
    print(f"График линзирования сохранен: {img_path.resolve()}")

    video_path = output_dir / "gravitational_lensing.mp4"
    start_time = time.time()
    try:
        vis.save(video_path, fps=30, duration=6.0)
        print(f"MP4-анимация сохранена: {video_path.resolve()} ({time.time() - start_time:.2f} сек)")
    except FileNotFoundError:
        gif_path = output_dir / "gravitational_lensing.gif"
        vis.save(gif_path, fps=25, duration=6.0)
        print(f"GIF-анимация сохранена: {gif_path.resolve()}")


if __name__ == "__main__":
    main()