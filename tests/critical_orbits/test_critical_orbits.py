"""Тестовый сценарий: поведение лучей в критической зоне около фотонной сферы."""

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


def run_critical_batch(rs: float) -> List[PhotonSimulation]:
    """Генерирует плотный пучок фотонов в окрестности b_crit."""
    b_crit = np.sqrt(27.0) / 2.0 * rs
    offsets = np.concatenate([
        np.linspace(-0.25, -0.005, 12),
        np.linspace(-0.003, -0.0002, 6),
        np.linspace(0.0002, 0.003, 6),
        np.linspace(0.005, 0.25, 12)
    ])
    impact_params = sorted([float(b_crit + delta) for delta in offsets])

    completed = []
    print(f"Запуск расчета критической зоны (b_crit ≈ {b_crit:.5f} rs)...")
    for b in impact_params:
        sim = PhotonSimulation(r_s=rs, b=b, phi_0=1e-3, phi_max=12.0 * np.pi)
        try:
            sim.run(rtol=1e-9, atol=1e-11, max_step=0.02)
            completed.append(sim)
            print(f"b = {b:8.5f} rs | {sim.status:<22} | r_min = {sim.r_min:.4f} rs")
        except (AssertionError, ValueError) as err:
            print(f"b = {b:8.5f} rs | Ошибка ({err}) — пропуск")

    return completed


def main() -> None:
    """Генерация детальной макро-визуализации фотонной сферы."""
    rs_val = 1.0
    output_dir = Path(current_dir)

    sims = run_critical_batch(rs=rs_val)
    full_beam = []
    for sim in sims:
        full_beam.append(sim)
        full_beam.append(MirroredPhoton(sim))

    vis = Visualizer(fig_size=(16, 9))
    vis.draw_black_hole(rs=rs_val)
    vis.plot_trajectories(full_beam, crop_box=6.0)
    vis.set_limits((-4.5, 4.5), (-2.53, 2.53))

    img_path = output_dir / "critical_orbits_zoom.png"
    vis.save(img_path, dpi=300)
    print(f"График сохранен: {img_path.resolve()}")

    video_path = output_dir / "critical_orbits_zoom.mp4"
    start_time = time.time()
    try:
        vis.save(video_path, fps=30, duration=8.0)
        print(f"MP4-анимация сохранена: {video_path.resolve()} ({time.time() - start_time:.2f} сек)")
    except FileNotFoundError:
        gif_path = output_dir / "critical_orbits_zoom.gif"
        vis.save(gif_path, fps=25, duration=8.0)
        print(f"GIF-анимация сохранена: {gif_path.resolve()}")


if __name__ == "__main__":
    main()