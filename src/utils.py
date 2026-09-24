"""Модуль вспомогательных утилит и конфигурации оформления графиков."""

from pathlib import Path
from typing import Dict, Any, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np

TrajectoryData = Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]

STYLE_CONFIG: Dict[str, Any] = {
    "bg_dark": "#050814",
    "bg_axes": "#080e21",
    "grid_color": "#16203d",
    "text_color": "#d6e2f0",
    "horizon_color": "#000000",
    "photon_sphere": "#f59e0b",
    "shadow_color": "#2563eb",
    "ray_color": "#f0f9ff",
    "ray_glow": "#38bdf8",
    "ray_escape_core": "#f0f9ff",
    "ray_escape_glow": "#38bdf8",
    "ray_capture_core": "#fffbeb",
    "ray_capture_glow": "#fde68a",
    "ray_critical_core": "#ffffff",
    "ray_critical_glow": "#93c5fd",
    "theory_line": "#c084fc",
    "numeric_points": "#38bdf8",
}


def apply_space_style() -> None:
    """Устанавливает глобальные параметры matplotlib rcParams для темной темы."""
    plt.rcParams.update({
        "figure.facecolor": STYLE_CONFIG["bg_dark"],
        "axes.facecolor": STYLE_CONFIG["bg_axes"],
        "axes.edgecolor": STYLE_CONFIG["grid_color"],
        "axes.labelcolor": STYLE_CONFIG["text_color"],
        "xtick.color": STYLE_CONFIG["text_color"],
        "ytick.color": STYLE_CONFIG["text_color"],
        "grid.color": STYLE_CONFIG["grid_color"],
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 10,
        "figure.autolayout": True,
    })


def ensure_output_dir(dir_path: str | Path) -> Path:
    """Проверяет существование директории для сохранения и создает её при отсутствии."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path