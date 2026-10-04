"""Render explanatory README figures; no experimental/model results are used.

Run from the repository root: python scripts/render_readme_figures.py
Dependencies: matplotlib and numpy. Outputs are deterministic SVG + PNG files.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


OUT = ROOT / "docs" / "assets"
INK, MUTED = "#172B40", "#42566C"
ORANGE = "#E69F00"


def configure_font():
    # Embed glyph paths in SVG so GitHub does not depend on the reader's fonts.
    windows_font = Path("C:/Windows/Fonts/msyh.ttc")
    if windows_font.is_file():
        font_manager.fontManager.addfont(str(windows_font))
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in names:
            return name
    raise RuntimeError("Install Microsoft YaHei or Noto Sans CJK SC to render Chinese labels")


def save(fig, name, description):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", facecolor="white", metadata={"Description": description, "Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white", metadata={"Description": description})
    plt.close(fig)


def mechanism():
    # Synthetic 16-band cube: fixed analytic shapes and spectra, no sampled data.
    y, x = np.mgrid[0:96, 0:128]
    object_mask = ((x - 67) / 22) ** 2 + ((y - 49) / 29) ** 2 <= 1
    band = np.linspace(0, 1, 16)
    background = 0.12 + 0.12 * band
    foreground = 0.36 + 0.35 * np.sin(np.pi * band) ** 2
    cube = np.broadcast_to(background, (96, 128, 16)).copy()
    cube[object_mask] = foreground
    rgb = cube[..., [10, 6, 2]]  # Fixed band assignment; same [0,1] scale in all panels.
    coarse = np.array([34., 14., 103., 87.])
    raw_refined = np.array([44., 19., 90., 79.])
    blended = coarse + 0.5 * (raw_refined - coarse)

    fig = plt.figure(figsize=(12, 3.25), facecolor="white")
    panels = [fig.add_axes([0.035 + i * 0.241, 0.22, 0.218, 0.61]) for i in range(4)]
    titles = ["(a) 输入", "(b) 候选框", "(c) 输出", "(d) 局部放大"]

    def box(ax, coords, color, dashed=False):
        x1, y1, x2, y2 = coords
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False,
                               edgecolor=color, linewidth=2.2,
                               linestyle=(0, (4, 2)) if dashed else "solid"))

    for i, (ax, title) in enumerate(zip(panels, titles)):
        ax.imshow(rgb, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, loc="left", fontsize=15, pad=11, color=INK)
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_edgecolor("#CBD5E1")
        if i >= 1:
            box(ax, coarse, ORANGE, True)
        if i >= 2:
            box(ax, blended, "#FFFFFF")
        if i == 2:
            ax.add_patch(Rectangle((29, 9), 31, 31, fill=False, edgecolor="#DDEAF3", linewidth=1))
        if i == 3:
            ax.set_xlim(29, 60)
            ax.set_ylim(40, 9)
    for i in range(3):
        fig.text(0.263 + i * 0.241, 0.52, "→", fontsize=20, color=MUTED, ha="center")
    fig.text(0.035, 0.08, "虚线：候选框     实线：输出框", fontsize=13, color=INK)
    save(fig, "refinement-concept", "Synthetic 16-band input with manually specified boxes; NOT model predictions or measured improvement. See docs/figures.md.")


def architecture():
    fig, ax = plt.subplots(figsize=(12.4, 4.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 1240), ylim=(420, 0))
    ax.axis("off")

    def panel(x, y, w, h, title, subtitle=None, fill="#F7F8FA"):
        ax.add_patch(Rectangle((x, y), w, h, linewidth=1, edgecolor="#687583", facecolor=fill))
        if subtitle:
            ax.text(x+w/2, y+h/2-11, title, ha="center", va="center", fontsize=15, color=INK)
            ax.text(x+w/2, y+h/2+16, subtitle, ha="center", va="center", fontsize=12, color=MUTED)
        else:
            ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=15, color=INK)

    def arrow(points, color=MUTED, dashed=False):
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.35,
                    linestyle="--" if dashed else "solid")
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                    mutation_scale=13, linewidth=1.35, color=color,
                                    linestyle="--" if dashed else "solid", shrinkA=0, shrinkB=0))

    panel(30, 190, 160, 75, "高光谱输入", "16 × H × W")
    panel(240, 55, 160, 70, "光谱适配", "1 × 1 Conv")
    panel(445, 55, 180, 70, "RF-DETR Small")
    panel(240, 300, 385, 70, "高分辨率 CNN", "stride 4")
    panel(735, 225, 145, 75, "ROI 融合", fill="#EDF3F8")
    panel(920, 225, 150, 75, "边界分布", "L / T / R / B", fill="#EDF3F8")
    panel(920, 335, 150, 65, "解码与融合", fill="#EDF3F8")
    panel(1100, 55, 120, 70, "分类 logits")
    panel(1100, 335, 120, 65, "检测框")

    arrow([(190, 212), (215, 212), (215, 90), (240, 90)])
    arrow([(190, 243), (215, 243), (215, 335), (240, 335)])
    arrow([(400, 90), (445, 90)])
    # Separate classification and box outputs avoid a perimeter bypass.
    arrow([(625, 77), (1100, 77)])
    arrow([(535, 125), (535, 248), (735, 248)])
    ax.text(605, 234, "全局特征", fontsize=12, color=MUTED)
    arrow([(625, 335), (685, 335), (685, 278), (735, 278)])
    ax.text(640, 363, "局部特征", fontsize=12, color=MUTED)
    arrow([(880, 262), (920, 262)])
    arrow([(995, 300), (995, 335)])
    arrow([(1070, 367), (1100, 367)])
    # Coarse boxes provide the shared ROI and the boundary prior.
    arrow([(625, 103), (680, 103), (680, 165), (995, 165), (995, 225)], dashed=True)
    ax.text(835, 152, "候选框", fontsize=12, color=MUTED, ha="center")
    arrow([(807, 165), (807, 225)], dashed=True)
    save(fig, "architecture", "Two feature branches feed shared ROI fusion and boundary refinement. Ground truth augments training candidates only; evaluation predictions are target-independent.")


if __name__ == "__main__":
    font = configure_font()
    with plt.rc_context({"font.family": font, "svg.fonttype": "path", "svg.hashsalt": "hifr-readme-v1"}):
        mechanism()
        architecture()
    print(f"Rendered README figures in {OUT}")
