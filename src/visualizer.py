"""Модуль визуализации черной дыры и траекторий фотонов."""

from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import numpy as np

from utils import STYLE_CONFIG, apply_space_style, ensure_output_dir


class Visualizer:
    """Класс для построения графиков и сохранения результатов моделирования."""

    def __init__(self, fig_size: Tuple[int, int] = (16, 9)) -> None:
        """Инициализирует полотно визуализации с заданными пропорциями."""
        apply_space_style()
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self._trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
        self._static_lines: List[Line2D] = []
        self._setup_canvas()

    def _setup_canvas(self) -> None:
        """Настраивает сетку, равный масштаб осей и базовую разметку."""
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X [r_s]")
        self.ax.set_ylabel("Y [r_s]")
        self.ax.set_title(
            "Траектории фотонов вблизи горизонта событий",
            color=STYLE_CONFIG["text_color"],
            pad=15,
            fontsize=12,
            fontweight="bold"
        )

    def draw_black_hole(self, rs: float = 1.0) -> None:
        """Отрисовывает черную дыру как сплошной черный диск без вспомогательных линий."""
        black_hole = patches.Circle(
            (0, 0),
            rs,
            facecolor=STYLE_CONFIG["horizon_color"],
            edgecolor="none",
            zorder=5,
            label=f"Черная дыра ($r_s$)"
        )
        self.ax.add_patch(black_hole)

    def plot_trajectories(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        labels: Optional[List[str]] = None
    ) -> None:
        """Отображает набор переданных траекторий фотонов с неоновым шлейфом."""
        self._trajectories = list(trajectories)

        for idx, (x, y) in enumerate(trajectories):
            label = labels[idx] if labels and idx < len(labels) else None

            # Внешний полупрозрачный неоновый ореол
            g_line, = self.ax.plot(
                x, y,
                color=STYLE_CONFIG["ray_glow"],
                linewidth=3.0,
                alpha=0.25,
                zorder=3
            )
            # Центральный белый луч
            c_line, = self.ax.plot(
                x, y,
                color=STYLE_CONFIG["ray_color"],
                linewidth=1.0,
                alpha=0.95,
                zorder=4,
                label=label
            )
            self._static_lines.extend([g_line, c_line])

    def set_limits(self, x_limit: Tuple[float, float], y_limit: Tuple[float, float]) -> None:
        """Задает видимые границы координатной сетки."""
        self.ax.set_xlim(x_limit)
        self.ax.set_ylim(y_limit)

    def finalize(self) -> None:
        """Размещает легенду (если есть элементы с метками) и настраивает стили."""
        handles, _ = self.ax.get_legend_handles_labels()
        if handles:
            legend = self.ax.legend(
                loc="upper right",
                facecolor=STYLE_CONFIG["bg_axes"],
                edgecolor=STYLE_CONFIG["grid_color"],
                labelcolor=STYLE_CONFIG["text_color"],
                framealpha=0.85
            )
            legend.get_frame().set_linewidth(0.8)

    def _save_animation(
        self,
        out_path: Path,
        dpi: int,
        fps: int,
        frames: Optional[int]
    ) -> None:
        """Внутренний метод генерации покадровой анимации движения фотонов."""
        if not self._trajectories:
            raise ValueError(
                "Нет данных траекторий для создания анимации. "
                "Сначала передайте координаты через plot_trajectories()."
            )

        for line in self._static_lines:
            line.set_visible(False)

        max_pts = max((len(x) for x, _ in self._trajectories), default=100)
        num_frames = frames if frames is not None else min(max(max_pts, 60), 120)

        anim_artists = []
        for _ in self._trajectories:
            glow_trail, = self.ax.plot(
                [], [], color=STYLE_CONFIG["ray_glow"], linewidth=3.0, alpha=0.25, zorder=3
            )
            core_trail, = self.ax.plot(
                [], [], color=STYLE_CONFIG["ray_color"], linewidth=1.0, alpha=0.95, zorder=4
            )
            head_glow, = self.ax.plot(
                [], [], marker="o", markersize=7, color=STYLE_CONFIG["ray_glow"], alpha=0.6, zorder=6
            )
            head_core, = self.ax.plot(
                [], [], marker="o", markersize=3.5, color="#ffffff", alpha=1.0, zorder=7
            )
            anim_artists.append({
                "glow_trail": glow_trail,
                "core_trail": core_trail,
                "head_glow": head_glow,
                "head_core": head_core
            })

        def update(frame_idx: int):
            t = frame_idx / max(1, num_frames - 1)
            updated_objects = []
            for (x, y), arts in zip(self._trajectories, anim_artists):
                n_pts = len(x)
                if n_pts == 0:
                    continue
                curr_idx = min(int(t * n_pts), n_pts - 1)

                x_sub = x[:curr_idx + 1]
                y_sub = y[:curr_idx + 1]

                arts["glow_trail"].set_data(x_sub, y_sub)
                arts["core_trail"].set_data(x_sub, y_sub)
                arts["head_glow"].set_data([x[curr_idx]], [y[curr_idx]])
                arts["head_core"].set_data([x[curr_idx]], [y[curr_idx]])
                updated_objects.extend(arts.values())
            return updated_objects

        anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=num_frames,
            interval=1000 // fps,
            blit=False
        )

        ext = out_path.suffix.lower()
        if ext == ".gif":
            writer = animation.PillowWriter(fps=fps)
            anim.save(out_path, writer=writer, dpi=dpi)
        elif ext == ".mp4":
            writer = animation.FFMpegWriter(fps=fps, codec="h264")
            anim.save(out_path, writer=writer, dpi=dpi)
        else:
            raise ValueError(f"Неподдерживаемый формат анимации: {ext}")

        for arts in anim_artists:
            for art in arts.values():
                art.remove()
        for line in self._static_lines:
            line.set_visible(True)

    def save(
        self,
        filepath: str | Path,
        dpi: Optional[int] = None,
        fps: int = 30,
        frames: Optional[int] = None
    ) -> None:
        """Сохраняет результат.

        Автоматически определяет формат по расширению:
        - '.gif', '.mp4' -> покадровая анимация движения фотонов.
        - '.png', '.jpg', '.pdf' -> статическое изображение высокого качества.
        """
        out_path = Path(filepath)
        ensure_output_dir(out_path.parent)
        self.finalize()

        ext = out_path.suffix.lower()
        if ext in (".gif", ".mp4"):
            target_dpi = dpi if dpi is not None else 120
            self._save_animation(out_path, dpi=target_dpi, fps=fps, frames=frames)
        else:
            target_dpi = dpi if dpi is not None else 400
            self.fig.savefig(out_path, dpi=target_dpi, facecolor=self.fig.get_facecolor())

    def show(self) -> None:
        """Отображает окно графика."""
        self.finalize()
        plt.show()