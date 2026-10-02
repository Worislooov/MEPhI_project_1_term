"""Главный исполнительный модуль симуляции и визуализации траекторий фотонов."""

from pathlib import Path
import time
from typing import List, Tuple
import numpy as np

from simulation import PhotonSimulation
from visualizer import Visualizer


class MirroredPhoton:
    """Зеркальная копия фотона относительно оси X для построения симметричного пучка."""

    def __init__(self, sim: PhotonSimulation) -> None:
        self.x = sim.x
        self.y = -sim.y
        self.phi = sim.phi
        self.status = sim.status
        self.b = sim.b
        self.r_s = sim.r_s
        self.r_min = sim.r_min


def run_photon_batch(
    rs: float,
    impact_parameters: List[float]
) -> List[PhotonSimulation]:
    """Запускает численное моделирование для положительных прицельных параметров."""
    completed_sims = []
    print(f"Запуск численного расчета для {len(impact_parameters)} фотонов...")

    for b in impact_parameters:
        sim = PhotonSimulation(r_s=rs, b=b, phi_0=1e-3, phi_max=6.0 * np.pi)
        try:
            sim.run(rtol=1e-8, atol=1e-10, max_step=0.04)
            completed_sims.append(sim)
            print(f"b = {b:5.2f} rs | Исход: {sim.status:<22} | r_min = {sim.r_min:.3f} rs")
        except (AssertionError, ValueError) as err:
            sim.status = "Ошибка расчета"
            print(f"b = {b:5.2f} rs | Исход: Ошибка ({err}) — луч пропущен")

    return completed_sims


def build_symmetric_beam(
    simulations: List[PhotonSimulation]
) -> List[object]:
    """Объединяет исходные и зеркальные лучи в единый симметричный фронт."""
    full_beam = []
    for sim in simulations:
        full_beam.append(sim)
        full_beam.append(MirroredPhoton(sim))
    return full_beam


def extract_deflection_data(
    simulations: List[PhotonSimulation]
) -> Tuple[np.ndarray, np.ndarray]:
    """Извлекает угол отклонения alpha = phi_exit - pi для пролетных лучей."""
    b_vals = []
    alpha_vals = []

    for sim in simulations:
        if sim.status != "В черной дыре" and sim.phi is not None and len(sim.phi) > 0:
            phi_exit = float(sim.phi[-1])
            if phi_exit > np.pi:
                b_vals.append(sim.b)
                alpha_vals.append(phi_exit - np.pi)

    return np.array(b_vals), np.array(alpha_vals)


def main() -> None:
    """Точка входа: расчет широкополосного пучка и сохранение графики в MP4 и PNG."""
    rs_val = 1.0

    impact_params = [float(val) for val in np.linspace(0.00001, 9, 100)]

    start_calc = time.time()
    successful_sims = run_photon_batch(rs=rs_val, impact_parameters=impact_params)
    print(f"Расчет завершен за {time.time() - start_calc:.2f} сек.")

    all_rays = build_symmetric_beam(successful_sims)
    print(f"Сформирован симметричный пучок из {len(all_rays)} лучей.\n")

    vis_scene = Visualizer(fig_size=(16, 9))
    vis_scene.draw_black_hole(rs=rs_val)
    vis_scene.plot_trajectories(all_rays, crop_box=20.0)
    vis_scene.set_limits((-16.0, 16.0), (-9.125, 9.125))

    img_output = Path("results") / "main_pic.png"
    vis_scene.save(img_output, dpi=300)
    print(f"Статичный график сохранен: {img_output.resolve()}")

    video_output = Path("results") / "main_vid.mp4"
    start_anim = time.time()
    try:
        vis_scene.save(video_output, fps=30, duration=10.0)
        print(f"MP4-анимация сохранена: {video_output.resolve()} ({time.time() - start_anim:.2f} сек)")
    except FileNotFoundError as err:
        print(f"Ошибка сохранения MP4: {err}")
        fallback_gif = Path("results") / "main_gif.gif"
        vis_scene.save(fallback_gif, fps=25, duration=4.0)
        print(f"Анимация сохранена в GIF: {fallback_gif.resolve()}")

    b_num, alpha_num = extract_deflection_data(successful_sims)
    if len(b_num) > 0:
        vis_comparison = Visualizer(fig_size=(10, 6))
        vis_comparison.plot_comparison(
            b_numerical=b_num,
            alpha_numerical=alpha_num,
            rs=rs_val
        )
        comp_output = Path("results") / "deflection_angle.png"
        vis_comparison.save(comp_output, dpi=300)
        print(f"Сравнительный график сохранен: {comp_output.resolve()}")


if __name__ == "__main__":
    main()