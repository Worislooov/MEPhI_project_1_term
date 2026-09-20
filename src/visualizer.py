"""Модуль визуализации черной дыры и траекторий фотонов."""

from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from utils import STYLE_CONFIG, apply_space_style, ensure_output_dir


class Visualizer:
    """Класс для построения графиков и сохранения результатов моделирования."""

    def __init__(self, fig_size: Tuple[int, int] = (16, 9)) -> None:
        """Инициализирует полотно визуализации с заданными пропорциями."""
        apply_space_style()
        self.fig, self.ax = plt.subplots(figsize=fig_size)
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
            zorder=5
        )
        self.ax.add_patch(black_hole)

    def plot_trajectories(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]],
        labels: Optional[List[str]] = None
    ) -> None:
        """Отображает набор переданных траекторий фотонов с неоновым шлейфом."""
        for idx, (x, y) in enumerate(trajectories):
            label = labels[idx] if labels and idx < len(labels) else None

            self.ax.plot(
                x, y,
                color=STYLE_CONFIG["ray_glow"],
                linewidth=3.0,
                alpha=0.25,
                zorder=3
            )
            self.ax.plot(
                x, y,
                color=STYLE_CONFIG["ray_color"],
                linewidth=1.0,
                alpha=0.95,
                zorder=4,
                label=label
            )

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

    def save(self, filepath: str | Path, dpi: int = 400) -> None:
        """Сохраняет изображение графика в заданном разрешении (стандарт — 400 DPI)."""
        out_path = Path(filepath)
        ensure_output_dir(out_path.parent)
        self.finalize()
        self.fig.savefig(out_path, dpi=dpi, facecolor=self.fig.get_facecolor())

    def show(self) -> None:
        """Отображает окно графика."""
        self.finalize()
        plt.show()