"""Publication style for the v5 figures.

Implements the constant and helper API of the scientific-figure-making skill:
PALETTE, FigureStyle, apply_publication_style, create_subplots,
finalize_figure, make_grouped_bar, annotate_bars, make_trend, make_heatmap,
make_scatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}
DEFAULT_COLORS = [PALETTE["blue_main"], PALETTE["green_3"],
                  PALETTE["red_strong"], PALETTE["teal"],
                  PALETTE["violet"], PALETTE["neutral"]]


@dataclass(frozen=True)
class FigureStyle:
    font_size: int = 16
    axes_linewidth: float = 2.5
    use_tex: bool = False
    font_family: tuple = ("DejaVu Sans", "Helvetica", "Arial", "sans-serif")


def apply_publication_style(style: FigureStyle | None = None) -> None:
    style = style or FigureStyle()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": list(style.font_family),
        "font.size": style.font_size,
        "axes.labelsize": style.font_size,
        "axes.titlesize": style.font_size,
        "legend.fontsize": max(5, style.font_size - 1),
        "xtick.labelsize": max(5, style.font_size - 1),
        "ytick.labelsize": max(5, style.font_size - 1),
        "axes.linewidth": style.axes_linewidth,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "text.usetex": style.use_tex,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def create_subplots(nrows=1, ncols=1, figsize=None, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    axes = np.atleast_1d(axes).ravel()
    return fig, axes


def finalize_figure(fig, out_path, formats=None, dpi=300, close=True,
                    pad=0.05, **kwargs):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if formats is None:
        formats = [out_path.suffix.lstrip(".") or "pdf"]
    saved = []
    for fmt in formats:
        path = out_path.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    pad_inches=pad, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def make_grouped_bar(ax, categories, series, labels, ylabel="Value",
                     colors=None, annotate=False):
    categories = list(categories)
    series = [np.asarray(s) for s in series]
    if any(len(s) != len(categories) for s in series):
        raise ValueError("each series must match the number of categories")
    colors = colors or DEFAULT_COLORS
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    last = None
    for i, (vals, label) in enumerate(zip(series, labels)):
        last = ax.bar(x + (i - (len(series) - 1) / 2) * width, vals, width,
                      label=label, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    if annotate and last is not None:
        annotate_bars(ax, last)
    return last


def annotate_bars(ax, bars, fmt="{:.2f}", fontsize=10, padding=3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, padding), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize)


def make_trend(ax, x, y_series, labels, colors=None, ylabel=None,
               xlabel=None, show_shadow=True):
    x = np.asarray(x)
    colors = colors or DEFAULT_COLORS
    for i, (y, label) in enumerate(zip(y_series, labels)):
        y = np.asarray(y)
        if len(y) != len(x):
            raise ValueError("each series must match x")
        ax.plot(x, y, label=label, color=colors[i % len(colors)])
        if show_shadow and y.ndim == 2:
            ax.fill_between(x, y.min(axis=1), y.max(axis=1),
                            color=colors[i % len(colors)], alpha=0.16,
                            linewidth=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if labels:
        ax.legend()


def make_heatmap(ax, matrix, x_labels=None, y_labels=None, cmap="magma",
                 cbar_label=None, annotate=False):
    matrix = np.asarray(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
    if cbar_label:
        ax.figure.colorbar(im, ax=ax, label=cbar_label)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                        va="center", fontsize=6)
    return im


def make_scatter(ax, x, y, label=None, color=None, size=50, alpha=0.7):
    ax.scatter(x, y, s=size, alpha=alpha, label=label,
               color=color or PALETTE["blue_main"])
    if label:
        ax.legend()
