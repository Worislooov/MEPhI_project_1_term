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
        self._trajectories: List[Dict[str, Any]] = []
        self._static_lines: List[Line2D] = []
        self._setup_canvas()

    def _setup_canvas(self) -> None:
        """Настраивает сетку, равный масштаб осей и базовую разметку."""
        self.ax.set_aspect("equal")
        self.ax.grid(True)
        self.ax.set_xlabel("X [$r_s$]")
        self.ax.set_ylabel("Y [$r_s$]")
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

    def _resolve_ray_colors(self, ray_type: str) -> Tuple[str, str]:
        """Определяет цветовую температуру луча в зависимости от характера траектории."""
        if ray_type == "capture":
            return STYLE_CONFIG["ray_capture_core"], STYLE_CONFIG["ray_capture_glow"]
        if ray_type == "critical":
            return STYLE_CONFIG["ray_critical_core"], STYLE_CONFIG["ray_critical_glow"]
        return STYLE_CONFIG["ray_escape_core"], STYLE_CONFIG["ray_escape_glow"]

    def _extract_trajectory_data(
        self,
        item: Any,
        crop_box: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
        """Извлекает координаты, время и статус из объекта симуляции или кортежа."""
        ray_type = "escape"
        t_arr = None

        if hasattr(item, "x") and hasattr(item, "y"):
            x_arr = np.asarray(item.x, dtype=float)
            y_arr = np.asarray(item.y, dtype=float)
            status = getattr(item, "status", "")
            if status == "В черной дыре":
                ray_type = "capture"
            elif hasattr(item, "b") and hasattr(item, "r_s"):
                if abs(abs(item.b) - np.sqrt(27.0) / 2.0 * item.r_s) < 0.05:
                    ray_type = "critical"
        elif len(item) == 3:
            x_arr = np.asarray(item[0], dtype=float)
            y_arr = np.asarray(item[1], dtype=float)
            t_arr = np.asarray(item[2], dtype=float)
        else:
            x_arr = np.asarray(item[0], dtype=float)
            y_arr = np.asarray(item[1], dtype=float)

        if crop_box is not None and len(x_arr) > 0:
            mask = (np.abs(x_arr) <= crop_box) & (np.abs(y_arr) <= crop_box)
            if np.any(mask):
                first_idx = int(np.argmax(mask))
                last_idx = len(mask) - int(np.argmax(mask[::-1]))
                x_arr = x_arr[first_idx:last_idx]
                y_arr = y_arr[first_idx:last_idx]
                if t_arr is not None:
                    t_arr = t_arr[first_idx:last_idx]

        return x_arr, y_arr, t_arr, ray_type

    def plot_trajectories(
        self,
        trajectories: List[Any],
        ray_types: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        crop_box: Optional[float] = 12.0
    ) -> None:
        """Отображает набор переданных траекторий фотонов с неоновым шлейфом."""
        self._trajectories = []

        for idx, item in enumerate(trajectories):
            x_arr, y_arr, t_arr, inferred_type = self._extract_trajectory_data(item, crop_box)

            if len(x_arr) == 0 or len(y_arr) == 0:
                continue

            r_type = ray_types[idx] if ray_types and idx < len(ray_types) else inferred_type
            c_core, c_glow = self._resolve_ray_colors(r_type)

            self._trajectories.append({
                "x": x_arr,
                "y": y_arr,
                "t": t_arr,
                "type": r_type,
                "c_core": c_core,
                "c_glow": c_glow
            })

            label = labels[idx] if labels and idx < len(labels) else None

            g_line, = self.ax.plot(
                x_arr, y_arr,
                color=c_glow,
                linewidth=3.0,
                alpha=0.3,
                zorder=3
            )
            c_line, = self.ax.plot(
                x_arr, y_arr,
                color=c_core,
                linewidth=1.0,
                alpha=0.95,
                zorder=4,
                label=label
            )
            self._static_lines.extend([g_line, c_line])

    def plot_comparison(
        self,
        b_numerical: np.ndarray,
        alpha_numerical: np.ndarray,
        rs: float = 1.0,
        b_theory_min: Optional[float] = None,
        b_theory_max: Optional[float] = None
    ) -> None:
        """Строит график сопоставления численного расчета с теоретической формулой alpha = 2rs / b."""
        self.ax.clear()
        self.ax.set_aspect("auto")
        self.ax.grid(True)

        b_num = np.asarray(b_numerical, dtype=float)
        alpha_num = np.asarray(alpha_numerical, dtype=float)

        min_b = b_theory_min if b_theory_min is not None else max(float(np.min(b_num)), 2.6 * rs)
        max_b = b_theory_max if b_theory_max is not None else float(np.max(b_num))

        b_theory = np.linspace(min_b, max_b, 200)
        alpha_theory = (2.0 * rs) / b_theory

        self.ax.plot(
            b_theory, alpha_theory,
            color=STYLE_CONFIG["theory_line"],
            linewidth=2.0,
            linestyle="--",
            label=r"Теория ($\alpha = 2r_s / b$)"
        )
        self.ax.scatter(
            b_num, alpha_num,
            color=STYLE_CONFIG["numeric_points"],
            s=35,
            edgecolors=STYLE_CONFIG["ray_escape_core"],
            linewidths=0.8,
            zorder=5,
            label="Численное решение"
        )
        self.ax.plot(
            b_num, alpha_num,
            color=STYLE_CONFIG["numeric_points"],
            linewidth=1.0,
            alpha=0.4
        )

        self.ax.set_xlabel("Прицельный параметр $b$ [$r_s$]")
        self.ax.set_ylabel("Угол отклонения $\\alpha$ [рад]")
        self.ax.set_title(
            "Зависимость угла отклонения фотона от прицельного параметра",
            color=STYLE_CONFIG["text_color"],
            pad=15,
            fontsize=12,
            fontweight="bold"
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
                    "max_param": float(tr["t"][-1]),
                    "c_core": tr["c_core"],
                    "c_glow": tr["c_glow"]
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
                    "max_param": float(s[-1]),
                    "c_core": tr["c_core"],
                    "c_glow": tr["c_glow"]
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
        for p in path_data:
            glow_trail, = self.ax.plot(
                [], [], color=p["c_glow"], linewidth=3.0, alpha=0.3, zorder=3, animated=True
            )
            core_trail, = self.ax.plot(
                [], [], color=p["c_core"], linewidth=1.0, alpha=0.95, zorder=4, animated=True
            )
            head_glow, = self.ax.plot(
                [], [], marker="o", markersize=6, color=p["c_glow"], alpha=0.6, zorder=6, animated=True
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