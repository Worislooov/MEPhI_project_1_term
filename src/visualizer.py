"""Модуль визуализации черной дыры и траекторий фотонов."""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import numpy as np

from utils import STYLE_CONFIG, apply_space_style, ensure_output_dir, TrajectoryData



class Visualizer:
    """Класс для построения графиков и сохранения результатов моделирования."""

    def __init__(self, fig_size: Tuple[int, int] = (16, 9)) -> None:
        """Инициализирует полотно визуализации с заданными пропорциями."""
        apply_space_style()
        self.fig, self.ax = plt.subplots(figsize=fig_size)
        self._trajectories: List[Dict[str, np.ndarray]] = []
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
        trajectories: List[TrajectoryData],
        labels: Optional[List[str]] = None
    ) -> None:
        """Отображает набор переданных траекторий фотонов с неоновым шлейфом."""
        self._trajectories = []

        for item in trajectories:
            if len(item) == 3:
                x_raw, y_raw, t_raw = item
                x_arr = np.asarray(x_raw, dtype=float)
                y_arr = np.asarray(y_raw, dtype=float)
                t_arr = np.asarray(t_raw, dtype=float)
            else:
                x_raw, y_raw = item
                x_arr = np.asarray(x_raw, dtype=float)
                y_arr = np.asarray(y_raw, dtype=float)
                t_arr = None

            if len(x_arr) > 0 and len(y_arr) > 0:
                self._trajectories.append({
                    "x": x_arr,
                    "y": y_arr,
                    "t": t_arr
                })

        for idx, tr in enumerate(self._trajectories):
            label = labels[idx] if labels and idx < len(labels) else None

            g_line, = self.ax.plot(
                tr["x"], tr["y"],
                color=STYLE_CONFIG["ray_glow"],
                linewidth=3.0,
                alpha=0.25,
                zorder=3
            )
            c_line, = self.ax.plot(
                tr["x"], tr["y"],
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

    def _prepare_animation_data(self) -> Tuple[List[Dict[str, Any]], float]:
        """Формирует данные параметризации по времени t или по длине дуги."""
        has_time = all(tr["t"] is not None for tr in self._trajectories)
        prepared = []

        if has_time:
            global_max_param = max(float(tr["t"][-1]) for tr in self._trajectories)
            for tr in self._trajectories:
                prepared.append({
                    "x": tr["x"],
                    "y": tr["y"],
                    "param": tr["t"],
                    "max_param": float(tr["t"][-1])
                })
        else:
            for tr in self._trajectories:
                dx = np.diff(tr["x"])
                dy = np.diff(tr["y"])
                dists = np.hypot(dx, dy)
                s = np.concatenate(([0.0], np.cumsum(dists)))
                prepared.append({
                    "x": tr["x"],
                    "y": tr["y"],
                    "param": s,
                    "max_param": float(s[-1])
                })
            global_max_param = max((p["max_param"] for p in prepared), default=1.0)

        if global_max_param == 0.0:
            global_max_param = 1.0

        return prepared, global_max_param

    def _save_animation(
        self,
        out_path: Path,
        dpi: int,
        fps: int,
        duration: float
    ) -> None:
        """Создает анимацию движения фотонов с поддержкой шкалы времени и blitting."""
        if not self._trajectories:
            raise ValueError(
                "Нет данных траекторий для создания анимации. "
                "Сначала вызовите plot_trajectories()."
            )

        ext = out_path.suffix.lower()
        if ext == ".mp4" and not animation.FFMpegWriter.isAvailable():
            raise FileNotFoundError(
                "Утилита FFmpeg не найдена в системе. Установите её "
                "(например, winget install Gyan.FFmpeg) "
                "или сохраняйте файл с расширением .gif"
            )

        for line in self._static_lines:
            line.set_visible(False)

        path_data, max_param_val = self._prepare_animation_data()
        actual_fps = min(fps, 25) if ext == ".gif" else fps
        num_frames = max(int(actual_fps * duration), 25)

        anim_artists = []
        flat_artists = []
        for _ in self._trajectories:
            glow_trail, = self.ax.plot(
                [], [], color=STYLE_CONFIG["ray_glow"], linewidth=3.0, alpha=0.25, zorder=3, animated=True
            )
            core_trail, = self.ax.plot(
                [], [], color=STYLE_CONFIG["ray_color"], linewidth=1.0, alpha=0.95, zorder=4, animated=True
            )
            head_glow, = self.ax.plot(
                [], [], marker="o", markersize=6, color=STYLE_CONFIG["ray_glow"], alpha=0.6, zorder=6, animated=True
            )
            head_core, = self.ax.plot(
                [], [], marker="o", markersize=3, color="#ffffff", alpha=1.0, zorder=7, animated=True
            )
            entry = {
                "glow": glow_trail,
                "core": core_trail,
                "h_glow": head_glow,
                "h_core": head_core
            }
            anim_artists.append(entry)
            flat_artists.extend(entry.values())

        def init():
            for art in flat_artists:
                art.set_data([], [])
            return flat_artists

        def update(frame_idx: int):
            progress = frame_idx / max(1, num_frames - 1)
            current_val = progress * max_param_val

            for p, arts in zip(path_data, anim_artists):
                param_arr = p["param"]
                x = p["x"]
                y = p["y"]
                max_val = p["max_param"]

                if max_val == 0.0:
                    continue

                if current_val <= max_val:
                    curr_x = float(np.interp(current_val, param_arr, x))
                    curr_y = float(np.interp(current_val, param_arr, y))

                    idx = int(np.searchsorted(param_arr, current_val, side="right"))
                    sub_x = np.append(x[:idx], curr_x)
                    sub_y = np.append(y[:idx], curr_y)

                    arts["glow"].set_data(sub_x, sub_y)
                    arts["core"].set_data(sub_x, sub_y)
                    arts["h_glow"].set_data([curr_x], [curr_y])
                    arts["h_core"].set_data([curr_x], [curr_y])
                else:
                    arts["glow"].set_data(x, y)
                    arts["core"].set_data(x, y)
                    arts["h_glow"].set_data([], [])
                    arts["h_core"].set_data([], [])

            return flat_artists

        anim = animation.FuncAnimation(
            self.fig,
            update,
            init_func=init,
            frames=num_frames,
            interval=1000 // actual_fps,
            blit=True
        )

        if ext == ".gif":
            writer = animation.PillowWriter(fps=actual_fps)
            anim.save(out_path, writer=writer, dpi=dpi)
        elif ext == ".mp4":
            writer = animation.FFMpegWriter(fps=actual_fps, codec="h264")
            anim.save(out_path, writer=writer, dpi=dpi)
        else:
            raise ValueError(f"Неподдерживаемый формат анимации: {ext}")

        for art in flat_artists:
            art.remove()
        for line in self._static_lines:
            line.set_visible(True)

    def save(
        self,
        filepath: str | Path,
        dpi: Optional[int] = None,
        fps: int = 25,
        duration: float = 3.0
    ) -> None:
        """Сохраняет изображение или анимацию в зависимости от расширения."""
        out_path = Path(filepath)
        ensure_output_dir(out_path.parent)
        self.finalize()

        ext = out_path.suffix.lower()
        if ext in (".gif", ".mp4"):
            target_dpi = dpi if dpi is not None else 65
            self._save_animation(out_path, dpi=target_dpi, fps=fps, duration=duration)
        else:
            target_dpi = dpi if dpi is not None else 400
            self.fig.savefig(out_path, dpi=target_dpi, facecolor=self.fig.get_facecolor())

    def show(self) -> None:
        """Отображает окно графика."""
        self.finalize()
        plt.show()