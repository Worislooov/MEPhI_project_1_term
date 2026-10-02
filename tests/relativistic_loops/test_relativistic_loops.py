"""Тестовый сценарий: релятивистские световые петли и гравитационное зеркало."""

from pathlib import Path
import time
import os
import sys
from typing import List, Tuple
import matplotlib.patches as patches
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


def find_b_for_exit_angle(
    rs: float,
    target_phi: float,
    iterations: int = 24
) -> float:
    """Находит прицельный параметр b для заданного конечного угла вылета методом бисекции."""
    b_crit = np.sqrt(27.0) / 2.0 * rs
    low = b_crit + 1e-7
    high = 4.5

    for _ in range(iterations):
        mid = (low + high) / 2.0
        sim = PhotonSimulation(r_s=rs, b=mid, phi_0=1e-3, phi_max=8.0 * np.pi)
        try:
            sim.run(rtol=1e-8, atol=1e-10, max_step=0.03)
            if sim.status == "В черной дыре":
                low = mid
            else:
                phi_exit = float(sim.phi[-1])
                if phi_exit > target_phi:
                    low = mid
                else:
                    high = mid
        except (AssertionError, ValueError):
            low = mid

    return (low + high) / 2.0


def calculate_loop_targets(rs: float) -> List[Tuple[PhotonSimulation, str]]:
    """Вычисляет серию характерных траекторий с заданными порядками отклонения."""
    targets = [
        (np.pi * 1.5, "Поворот на 90°"),
        (np.pi * 2.0, "Зеркало 180° (разворот назад)"),
        (np.pi * 3.0, "1 полный виток (360°)"),
        (np.pi * 4.0, "1.5 витка (540°)"),
    ]

    results = []
    print("Калибровка прицельных параметров для целочисленных порядков отклонения...")

    for target_phi, label in targets:
        b_val = find_b_for_exit_angle(rs, target_phi)
        sim = PhotonSimulation(r_s=rs, b=b_val, phi_0=1e-3, phi_max=8.0 * np.pi)
        sim.run(rtol=1e-9, atol=1e-11, max_step=0.02)
        print(f"{label:<30} | b = {b_val:8.6f} rs | Выходной phi = {sim.phi[-1] / np.pi:5.2f} π")
        results.append((sim, f"{label} (b={b_val:.4f})"))

    return results


def main() -> None:
    """Визуализация дискретных порядков намотки траекторий вокруг фотонной сферы."""
    rs_val = 1.0
    output_dir = Path(current_dir)

    classified_rays = calculate_loop_targets(rs=rs_val)
    sim_objects = [item[0] for item in classified_rays]
    labels = [item[1] for item in classified_rays]

    vis = Visualizer(fig_size=(16, 9))
    vis.draw_black_hole(rs=rs_val)

    photon_sphere = patches.Circle(
        (0, 0),
        1.5 * rs_val,
        fill=False,
        edgecolor="#f59e0b",
        linestyle=":",
        linewidth=1.2,
        alpha=0.75,
        zorder=4,
        label=f"Фотонная сфера (1.5 $r_s$)"
    )
    vis.ax.add_patch(photon_sphere)

    vis.plot_trajectories(sim_objects, labels=labels, crop_box=8.0)
    vis.set_limits((-6.0, 6.0), (-3.375, 3.375))

    img_path = output_dir / "relativistic_loops.png"
    vis.save(img_path, dpi=300)
    print(f"\nСтатичный график сохранен: {img_path.resolve()}")

    video_path = output_dir / "relativistic_loops.mp4"
    start_time = time.time()
    try:
        vis.save(video_path, fps=30, duration=10.0)
        print(f"MP4-анимация сохранена: {video_path.resolve()} ({time.time() - start_time:.2f} сек)")
    except FileNotFoundError:
        gif_path = output_dir / "relativistic_loops.gif"
        vis.save(gif_path, fps=25, duration=10.0)
        print(f"GIF-анимация сохранена: {gif_path.resolve()}")


if __name__ == "__main__":
    main()