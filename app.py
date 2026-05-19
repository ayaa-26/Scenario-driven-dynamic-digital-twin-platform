# =====================================================================
# app.py  –  UniSim-Design Supervision — Sulfuric Acid Plant
#
# Compatible with:
#   - model/four.py          : Furnace class (dynamic)
#   - model/convertisseur.py : JumeauNumeriqueConvertisseur (PDE/RK4)
#   - simulation/main_simulation.py : simuler_complet()
#
# v3 — Added DYNAMIC tab:
#   • Simulated time response (step +5% sulfur flow at t=60s)
#   • Spatial profiles per bed (T, tau, r)
#   • Furnace and boiler thermal profiles
#   • SO3 ppm balance and absorption efficiencies
# =====================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import streamlit as st
from datetime import datetime
import io
import sys
import os
import base64

# ── Add root directory to path ────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from four import (
    DEFAULT_T_AIR_C  as DEFAULT_T_AIR,
    DEFAULT_T_SOUFRE_C as DEFAULT_T_S,
    DEFAULT_F_SOUFRE_KGMIN as DEFAULT_S,
    DEFAULT_F_AIR_NM3H as DEFAULT_AIR,
    DEFAULT_BYPASS_PCT,
)
DEFAULT_RATIO = 1.0 - DEFAULT_BYPASS_PCT / 100.0

from chaudiere import T_TARGET_CONV
from main_simulation import simuler_complet
from turbo_train import TurboBlowerTrain
from drying_tower import DryingTower
from air_filter import AirFilter

# ── Import heat exchangers ─────────────────────────────────────────────────
from exchangers import (
    HPSuperheater1B, HotInterpassHX, ColdInterpassHX,
    Economizer3B,
    HP4AExchanger, LP4AExchanger, E4CExchanger, E4AExchanger,
)
from exchangers_page import render_page_exchangers

# ── Import tanks ───────────────────────────────────────────────────────────
from bac import TankSystem, FlowIn, FlowOut


# ══════════════════════════════════════════════════════════════════════
# UNISIM-DESIGN STYLE PALETTE
# ══════════════════════════════════════════════════════════════════════

BG_MAIN      = '#B0B0B0'
BG_PANEL     = '#C8C8C8'
BG_DARK      = '#3C3C3C'
BG_EQUIP     = '#D0D0D0'
BG_DATA      = '#000000'
FG_DATA      = '#00FF44'
FG_DATA2     = '#00CC88'
FG_LABEL     = '#CCCCCC'
FG_TAG       = '#AAFFAA'
BORDER_EQ    = '#606060'

C_GAZ_JAUNE  = "#DD8500"
C_AIR        = "#F5F6F4"
C_WATER_BLUE = '#3366CC'
C_STEAM_RED  = '#DD2222'
C_ACID_VIOLET= '#9933CC'
C_SULFUR_ORG = "#BCCD06"
C_SEA_WATER  = '#0099DD'
C_CONDENSAT  = '#00CCFF'

METAL_LIGHT  = '#E8E8E8'
METAL_MID    = '#B8B8B8'
METAL_DARK   = '#888888'
METAL_BORDER = '#606060'

# ── DYNAMIC tab graph palette (dark oscilloscope style) ────────
DYN_BG       = '#0E1117'
DYN_PANEL    = '#1A1D27'
DYN_GRID     = '#2A2D3A'
DYN_GREEN    = '#00FF88'
DYN_ORANGE   = '#FF6B35'
DYN_CYAN     = '#00D4FF'
DYN_YELLOW   = '#FFD700'
DYN_RED      = '#FF4455'
DYN_PURPLE   = '#BB88FF'
DYN_WHITE    = '#E8E8F0'
DYN_GRAY     = '#6A6D7A'

COLORS_LITS  = [DYN_GREEN, DYN_ORANGE, DYN_CYAN, DYN_YELLOW]
LABELS_LITS  = ['Bed 1', 'Bed 2', 'Bed 3', 'Bed 4']


# ══════════════════════════════════════════════════════════════════════
# GRAPHIC HELPERS — UNISIM STYLE
# ══════════════════════════════════════════════════════════════════════

def unisim_bg(fig, ax, xlim=(0, 24), ylim=(0, 11)):
    fig.patch.set_facecolor(BG_MAIN)
    ax.set_facecolor(BG_MAIN)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.03)


def draw_title_bar(ax, title, xlim=24, ylim=11):
    ax.add_patch(plt.Rectangle((0, ylim - 0.55), xlim, 0.55,
                               facecolor=BG_DARK, edgecolor='none', zorder=20))
    ax.text(xlim / 2, ylim - 0.27, title,
            color='white', fontsize=11, fontweight='bold',
            ha='center', va='center', zorder=21, fontfamily='monospace')
    for i, (lbl, col) in enumerate([('×', '#CC3333'), ('□', '#888'), ('─', '#888')]):
        bx = xlim - 0.55 * (i + 1)
        ax.add_patch(plt.Rectangle((bx - 0.22, ylim - 0.46), 0.44, 0.36,
                                   facecolor=col if i == 0 else '#555',
                                   edgecolor='#333', lw=0.5, zorder=22))
        ax.text(bx, ylim - 0.28, lbl, color='white', fontsize=7,
                ha='center', va='center', zorder=23)


def cylinder_3d(ax, cx, cy, w, h, label='', tag='', horizontal=False):
    if horizontal:
        n = 30
        for i in range(n):
            frac = i / n
            if frac < 0.5:
                t = frac * 2
                gray = METAL_DARK if t < 0.15 else tuple(
                    np.array(mcolors.to_rgb(METAL_DARK)) * (1 - t) +
                    np.array(mcolors.to_rgb(METAL_LIGHT)) * t)
            else:
                t = (frac - 0.5) * 2
                gray = tuple(
                    np.array(mcolors.to_rgb(METAL_LIGHT)) * (1 - t) +
                    np.array(mcolors.to_rgb(METAL_DARK)) * t)
            ax.add_patch(plt.Rectangle(
                (cx - w / 2, cy - h / 2 + h * i / n), w, h / n,
                color=gray, zorder=2))
        for xc in [cx - w / 2, cx + w / 2]:
            ax.add_patch(Ellipse((xc, cy), width=h * 0.35, height=h,
                                 facecolor=METAL_MID, edgecolor=METAL_BORDER,
                                 lw=1.5, zorder=3))
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   fill=False, edgecolor=METAL_BORDER, lw=2, zorder=4))
    else:
        n = 30
        for i in range(n):
            frac = i / n
            if frac < 0.15:
                gray = METAL_DARK
            elif frac < 0.5:
                t = (frac - 0.15) / 0.35
                gray = tuple(np.array(mcolors.to_rgb(METAL_DARK)) * (1 - t) +
                             np.array(mcolors.to_rgb(METAL_LIGHT)) * t)
            elif frac < 0.85:
                t = (frac - 0.5) / 0.35
                gray = tuple(np.array(mcolors.to_rgb(METAL_LIGHT)) * (1 - t) +
                             np.array(mcolors.to_rgb(METAL_MID)) * t)
            else:
                gray = METAL_DARK
            ax.add_patch(plt.Rectangle(
                (cx - w / 2, cy - h / 2 + h * i / n), w, h / n,
                color=gray, zorder=2))
        for yc in [cy - h / 2, cy + h / 2]:
            ax.add_patch(Ellipse((cx, yc), width=w, height=w * 0.25,
                                 facecolor=METAL_MID, edgecolor=METAL_BORDER,
                                 lw=1.2, zorder=3))
        ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   fill=False, edgecolor=METAL_BORDER, lw=2, zorder=4))
    ax.text(cx, cy + 0.06, label, color='#111111', fontsize=9, fontweight='bold',
            ha='center', va='center', zorder=6, fontfamily='monospace')
    ax.text(cx, cy - 0.18, tag, color='#333333', fontsize=7,
            ha='center', va='center', zorder=6, fontfamily='monospace')


def draw_absorption_tower_unisim(ax, cx, cy, w, h, title, tag):
    cylinder_3d(ax, cx, cy, w, h, horizontal=False)
    pack_y1 = cy - h / 2 + h * 0.12
    pack_y2 = cy + h / 2 - h * 0.20
    n_levels = 6
    for i in range(n_levels + 1):
        frac = i / n_levels
        yy = pack_y1 + (pack_y2 - pack_y1) * frac
        ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [yy, yy],
                color='#999999', lw=0.6, zorder=5, alpha=0.8)
    ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [pack_y1, pack_y2],
            color='#888888', lw=0.8, zorder=5, alpha=0.7)
    ax.plot([cx + w / 2 - 0.04, cx - w / 2 + 0.04], [pack_y1, pack_y2],
            color='#888888', lw=0.8, zorder=5, alpha=0.7)
    dist_y = cy + h / 2 - h * 0.25
    ax.plot([cx - w / 2 + 0.05, cx + w / 2 - 0.05], [dist_y, dist_y],
            color=C_ACID_VIOLET, lw=1.5, zorder=6)
    for xi in np.linspace(cx - w / 2 + 0.07, cx + w / 2 - 0.07, 4):
        ax.plot([xi, xi], [dist_y, dist_y - 0.07], color=C_ACID_VIOLET, lw=1.0, zorder=6)
    ax.text(cx, cy + h / 2 + 0.22, title, color='#111111', fontsize=8,
            fontweight='bold', ha='center', va='bottom', zorder=7, fontfamily='monospace')
    ax.text(cx, cy - h / 2 - 0.14, tag, color='#333333', fontsize=6.5,
            ha='center', va='top', zorder=7, fontfamily='monospace')


def draw_pump_unisim(ax, cx, cy, R=0.18, color='#DDDD00', label='', tag=''):
    ax.add_patch(Circle((cx, cy), R, facecolor=color, edgecolor='#444', lw=1.5, zorder=6))
    tri_r = R * 0.55
    pts = np.array([
        [cx - tri_r * 0.6, cy - tri_r * 0.5],
        [cx - tri_r * 0.6, cy + tri_r * 0.5],
        [cx + tri_r * 0.9, cy],
    ])
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor='#444', edgecolor='#222', lw=0.8, zorder=7))
    ax.plot([cx - R * 1.1, cx + R * 1.1], [cy - R - 0.04, cy - R - 0.04],
            color='#444', lw=2.0, zorder=6)
    if label:
        ax.text(cx, cy + R + 0.08, label, color='#111', fontsize=6.5,
                ha='center', va='bottom', fontfamily='monospace', zorder=8)
    if tag:
        ax.text(cx, cy - R - 0.12, tag, color='#333', fontsize=6,
                ha='center', va='top', fontfamily='monospace', zorder=8)


def draw_heat_exchanger_unisim(ax, cx, cy, w, h, label='', tag=''):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                               boxstyle="round,pad=0.04",
                               facecolor=METAL_MID, edgecolor=METAL_BORDER,
                               lw=2, zorder=3))
    for i in range(1, 4):
        yy = cy - h / 2 + h * i / 4
        ax.plot([cx - w / 2 + 0.06, cx + w / 2 - 0.06], [yy, yy],
                color='#888', lw=0.8, zorder=4, alpha=0.7)
    for i in [1, 2]:
        xx = cx - w / 2 + w * i / 3
        ax.plot([xx, xx], [cy - h / 2 + 0.04, cy + h / 2 - 0.04],
                color='#777', lw=1.0, zorder=4, alpha=0.6)
    ax.text(cx, cy + 0.05, label, color='#111', fontsize=8, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=6)
    ax.text(cx, cy - 0.18, tag, color='#333', fontsize=6.5,
            ha='center', va='center', fontfamily='monospace', zorder=6)


def draw_compressor_unisim(ax, cx, cy, R, title='', tag=''):
    ax.add_patch(Circle((cx, cy), R, facecolor=METAL_MID, edgecolor=METAL_BORDER,
                        lw=2.5, zorder=5))
    ri = R * 0.42
    ax.add_patch(Circle((cx, cy), ri, facecolor=BG_EQUIP, edgecolor=METAL_BORDER,
                        lw=1.5, zorder=6))
    angles = [150, 270, 30]
    pts = np.array([[cx + ri * 0.72 * np.cos(np.radians(a)),
                     cy + ri * 0.72 * np.sin(np.radians(a))] for a in angles])
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor='#778899',
                             edgecolor=METAL_BORDER, lw=1, zorder=7))
    for deg in range(0, 360, 60):
        a = np.radians(deg)
        x0, y0 = cx + ri * 0.50 * np.cos(a), cy + ri * 0.50 * np.sin(a)
        x1, y1 = cx + ri * 0.92 * np.cos(a), cy + ri * 0.92 * np.sin(a)
        ax.plot([x0, x1], [y0, y1], color='#8899AA', lw=1.2, zorder=7)
    ax.text(cx, cy - R - 0.20, title, color='#111', fontsize=8, fontweight='bold',
            ha='center', va='top', fontfamily='monospace', zorder=8)
    ax.text(cx, cy - R - 0.42, tag, color='#333', fontsize=6.5,
            ha='center', va='top', fontfamily='monospace', zorder=8)


def data_box(ax, x, y, lines, w=1.55, color=FG_DATA, unit_color='#88FFBB'):
    lh = 0.26
    h = len(lines) * lh + 0.10
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="square,pad=0",
                               facecolor=BG_DATA, edgecolor='#555', lw=1.0, zorder=15))
    for i, line in enumerate(lines):
        yy = y + h - (i + 1) * lh + 0.04
        if isinstance(line, tuple):
            if len(line) == 2:
                val, unit = line
                ax.text(x + w - 0.22, yy, val, color=color,
                        fontsize=8, fontweight='bold', ha='right', va='center',
                        fontfamily='monospace', zorder=16)
                ax.text(x + w - 0.06, yy, unit, color=unit_color,
                        fontsize=6.5, ha='right', va='center',
                        fontfamily='monospace', zorder=16)
            else:
                txt = line[0]
                ax.text(x + w / 2, yy, txt, color=color,
                        fontsize=8, fontweight='bold', ha='center', va='center',
                        fontfamily='monospace', zorder=16)
        else:
            ax.text(x + 0.08, yy, line, color=color,
                    fontsize=7.5, fontweight='bold', ha='left', va='center',
                    fontfamily='monospace', zorder=16)


def stream_tag(ax, x, y, name, color='#DDDDDD'):
    w, h = len(name) * 0.085 + 0.20, 0.22
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="square,pad=0.02",
                               facecolor='#CCCCCC', edgecolor='#888', lw=0.8, zorder=18))
    ax.text(x, y, name, color='#111111', fontsize=6.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=19)


def pipe(ax, x1, y1, x2, y2, color=C_GAZ_JAUNE, lw=2.5, arrow=True, tag=None,
         tag_pos=0.5):
    if x1 != x2 and y1 != y2:
        xm, ym = x2, y1
        ax.plot([x1, xm], [y1, ym], color=color, lw=lw, solid_capstyle='butt', zorder=10)
        ax.plot([xm, x2], [ym, y2], color=color, lw=lw, solid_capstyle='butt', zorder=10)
        if arrow:
            dx, dy = x2 - xm, y2 - ym
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.3:
                frac = 0.5
                ax.annotate("",
                    xy=(xm + dx * (frac + 0.01), ym + dy * (frac + 0.01)),
                    xytext=(xm + dx * (frac - 0.01), ym + dy * (frac - 0.01)),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.18",
                                   color=color, lw=lw), zorder=11)
    else:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle='butt', zorder=10)
        if arrow:
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.3:
                frac = tag_pos
                ax.annotate("",
                    xy=(x1 + dx * (frac + 0.005), y1 + dy * (frac + 0.005)),
                    xytext=(x1 + dx * (frac - 0.005), y1 + dy * (frac - 0.005)),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.18",
                                   color=color, lw=lw), zorder=11)
    if tag:
        tx = (x1 + x2) / 2
        ty = (y1 + y2) / 2
        stream_tag(ax, tx, ty + 0.14, tag)


def draw_valve_symbol(ax, cx, cy, r=0.12):
    ax.add_patch(Circle((cx, cy), r, facecolor='#CCCCCC', edgecolor='#555',
                        lw=1.2, zorder=12))
    for ang in [0, 90]:
        a = np.radians(ang)
        ax.plot([cx - r * 0.7 * np.cos(a), cx + r * 0.7 * np.cos(a)],
                [cy - r * 0.7 * np.sin(a), cy + r * 0.7 * np.sin(a)],
                color='#444', lw=1.2, zorder=13)


def draw_bed_unisim(ax, cx, cy, w, h, bed_num, t_in, t_out, conv_pct, dp_kpa=0.0):
    n = 20
    for i in range(n):
        frac = i / n
        gray_val = 0.60 + 0.25 * np.sin(np.pi * frac)
        ax.add_patch(plt.Rectangle(
            (cx - w / 2, cy - h / 2 + h * i / n), w, h / n,
            color=(gray_val, gray_val, gray_val * 0.9), zorder=2))
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                               fill=False, edgecolor=METAL_BORDER, lw=1.8, zorder=4))
    ax.plot([cx - w / 2 + 0.05, cx + w / 2 - 0.05], [cy - h / 3, cy + h / 3],
            color='#999', lw=0.7, zorder=5, alpha=0.6)
    ax.plot([cx + w / 2 - 0.05, cx - w / 2 + 0.05], [cy - h / 3, cy + h / 3],
            color='#999', lw=0.7, zorder=5, alpha=0.6)
    ax.text(cx, cy + h / 2 + 0.14, f"BED {bed_num}",
            color='#111', fontsize=7.5, fontweight='bold',
            ha='center', va='bottom', fontfamily='monospace', zorder=6)
    data_box(ax, cx - w / 2 - 1.70, cy - 0.30, [
        (f"{t_in:.2f}", "°C in"),
        (f"{t_out:.2f}", "°C out"),
        (f"X={conv_pct:.2f}", "%"),
        (f"ΔP={dp_kpa:.2f}", "kPa"),
    ], w=1.60)


def draw_tank_unisim(ax, cx, cy, w, h, label, tag, level_pct, T, density,
                     color_fill, alarm=False, extra=None):
    cylinder_3d(ax, cx, cy, w, h, horizontal=False)
    if alarm:
        ax.add_patch(plt.Rectangle((cx - w / 2 - 0.02, cy - h / 2 - 0.02),
                                   w + 0.04, h + 0.04,
                                   fill=False, edgecolor='#CC2222',
                                   lw=2.5, zorder=10, linestyle='--'))
    fill_h = max(0, min(1, level_pct / 100)) * h
    if fill_h > 0:
        ax.add_patch(plt.Rectangle((cx - w / 2 + 0.04, cy - h / 2 + 0.03),
                                   w - 0.08, fill_h - 0.03,
                                   facecolor=color_fill, alpha=0.65,
                                   edgecolor='none', zorder=5))
    lvl_y = cy - h / 2 + fill_h
    if 0 < fill_h < h:
        ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [lvl_y, lvl_y],
                color='white', lw=1.2, zorder=7, alpha=0.8)
    ax.text(cx, cy + 0.12, label, color='#111', fontsize=7.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=8)
    ax.text(cx, cy - 0.10, tag, color='#333', fontsize=6,
            ha='center', va='center', fontfamily='monospace', zorder=8)
    lines_data = [
        (f"{level_pct:.2f}", "% LVL"),
        (f"{T:.2f}", "°C"),
        (f"{density:.1f}", "kg/m³"),
    ]
    if extra:
        lines_data.extend(extra)
    data_box(ax, cx + w / 2 - 1.3, cy - len(lines_data) * 0.13 - 1.6,
             lines_data, w=1.60)


# ══════════════════════════════════════════════════════════════════════
# GRAPHIC HELPERS — DYNAMIC PAGE (dark oscilloscope style)
# ══════════════════════════════════════════════════════════════════════

def _dyn_style_ax(ax, title='', xlabel='', ylabel='', ylim=None):
    """Apply dark oscilloscope style to a matplotlib axis."""
    ax.set_facecolor(DYN_PANEL)
    ax.tick_params(colors=DYN_WHITE, labelsize=8)
    ax.xaxis.label.set_color(DYN_WHITE)
    ax.yaxis.label.set_color(DYN_WHITE)
    ax.title.set_color(DYN_GREEN)
    ax.spines['bottom'].set_color(DYN_GRID)
    ax.spines['left'].set_color(DYN_GRID)
    ax.spines['top'].set_color(DYN_GRID)
    ax.spines['right'].set_color(DYN_GRID)
    ax.grid(True, color=DYN_GRID, linewidth=0.6, alpha=0.8)
    if title:
        ax.set_title(title, fontsize=9, fontweight='bold',
                     fontfamily='monospace', color=DYN_GREEN, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, fontfamily='monospace')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, fontfamily='monospace')
    if ylim:
        ax.set_ylim(*ylim)


def _dyn_fig(nrows, ncols, figsize):
    """Create a figure with dark background for DYNAMIC graphs."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(DYN_BG)
    fig.subplots_adjust(hspace=0.45, wspace=0.35,
                        left=0.07, right=0.97, top=0.93, bottom=0.09)
    return fig, axes


def _add_step_line(ax, t_step, label=True):
    """Draw vertical line for the disturbance step."""
    ax.axvline(x=t_step, color=DYN_RED, lw=1.2, linestyle='--', alpha=0.8)
    if label:
        ax.text(t_step + 5, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                'Step\n+5% S', color=DYN_RED, fontsize=6.5,
                fontfamily='monospace', va='bottom')


def _fig_to_b64(fig, dpi=130):
    """Convert a matplotlib figure to base64 for st.markdown."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                pad_inches=0.05, facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def render_page_dynamique(results):
    """
    Generate and display all DYNAMIC page graphs.
    Expected in results:
        results['historique']         → dict with t, tau_global, T_out_lits, ...
        results['convertisseur']['profils_lits']  → list of 4 dicts {z, T_C, tau, r, dp}
        results['four']['z_profil'], results['four']['T_profil']
        results['chaudiere']['z_profil'], results['chaudiere']['T_gaz_profil'], ...
    """
    hist = results.get('historique', {})
    conv = results.get('convertisseur', {})
    profils = conv.get('profils_lits', [])
    four    = results.get('four', {})
    chaud   = results.get('chaudiere', {})

    t            = np.array(hist.get('t', []))
    t_step       = hist.get('t_echelon', 60.0)
    delta_S_pct  = hist.get('delta_S_pct', 5.0)

    # ── KPI summary banner ─────────────────────────────────────────
    st.markdown("""
    <div style="background:#1A1D27;border:1px solid #2A2D3A;border-radius:6px;
                padding:10px 18px;margin-bottom:12px;font-family:'Courier New',monospace;">
      <span style="color:#00FF88;font-weight:bold;font-size:13px;">
        DYNAMIC PAGE — TEMPORAL RESPONSE & SPATIAL PROFILES</span>
      <span style="color:#6A6D7A;font-size:11px;margin-left:20px;">
        Simulated disturbance: step +{:.0f}% sulfur flow at t = {:.0f} s</span>
    </div>
    """.format(delta_S_pct, t_step), unsafe_allow_html=True)

    if len(t) == 0:
        st.warning("No history available. Please run a simulation first.")
        return

    # ══════════════════════════════════════════════════════════════
    # SECTION 1 — Global temporal responses
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="color:#00FF88;font-family:\'Courier New\',monospace;'
        'font-size:12px;font-weight:bold;border-left:3px solid #00FF88;'
        'padding-left:10px;margin:10px 0 6px 0;">1 — GLOBAL TEMPORAL RESPONSES</div>',
        unsafe_allow_html=True)

    fig1, axes1 = _dyn_fig(2, 3, figsize=(18, 8))

    tau_global = np.array(hist.get('tau_global', []))
    T_four_t   = np.array(hist.get('T_four', []))
    steam_t    = np.array(hist.get('steam', []))
    power_t    = np.array(hist.get('power', []))
    S_t        = np.array(hist.get('S_flow', []))
    ppm_so3_t  = np.array(hist.get('ppm_so3', []))

    # Graph 1: global τ(t)
    ax = axes1[0, 0]
    ax.plot(t, tau_global, color=DYN_GREEN, lw=1.8)
    ax.fill_between(t, tau_global, alpha=0.12, color=DYN_GREEN)
    _dyn_style_ax(ax, title='Global converter conversion rate τ (%)',
                  xlabel='Time (s)', ylabel='Conversion rate (%)')
    _add_step_line(ax, t_step)

    # Graph 2: furnace outlet T(t)
    ax = axes1[0, 1]
    ax.plot(t, T_four_t, color=DYN_ORANGE, lw=1.8)
    ax.fill_between(t, T_four_t, alpha=0.10, color=DYN_ORANGE)
    _dyn_style_ax(ax, title='Furnace 401AF01 outlet T (°C)',
                  xlabel='Time (s)', ylabel='Temperature (°C)')
    _add_step_line(ax, t_step)

    # Graph 3: sulfur flow S(t)
    ax = axes1[0, 2]
    ax.plot(t, S_t, color=DYN_YELLOW, lw=1.8, drawstyle='steps-post')
    _dyn_style_ax(ax, title='Sulfur feed flow rate (kg/min)',
                  xlabel='Time (s)', ylabel='Flow rate (kg/min)')
    _add_step_line(ax, t_step, label=False)

    # Graph 4: steam produced
    ax = axes1[1, 0]
    ax.plot(t, steam_t, color=DYN_RED, lw=1.8)
    ax.fill_between(t, steam_t, alpha=0.12, color=DYN_RED)
    _dyn_style_ax(ax, title='Steam produced — 401AV01 (t/h)',
                  xlabel='Time (s)', ylabel='Steam flow (t/h)')
    _add_step_line(ax, t_step)

    # Graph 5: recovered power
    ax = axes1[1, 1]
    ax.plot(t, power_t, color=DYN_CYAN, lw=1.8)
    ax.fill_between(t, power_t, alpha=0.10, color=DYN_CYAN)
    _dyn_style_ax(ax, title='Boiler recovered power (MW)',
                  xlabel='Time (s)', ylabel='Power (MW)')
    _add_step_line(ax, t_step)

    # Graph 6: SO3 ppm stack
    ax = axes1[1, 2]
    ax.plot(t, ppm_so3_t, color=DYN_PURPLE, lw=1.8)
    ax.axhline(y=10.0, color=DYN_RED, lw=1.0, linestyle=':', alpha=0.7,
               label='Regulatory limit')
    ax.fill_between(t, ppm_so3_t, alpha=0.10, color=DYN_PURPLE)
    _dyn_style_ax(ax, title='Residual SO3 stack emission (ppm)',
                  xlabel='Time (s)', ylabel='SO3 (ppm)')
    ax.legend(fontsize=6.5, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
              labelcolor=DYN_WHITE, loc='upper right')
    _add_step_line(ax, t_step)

    b64_1 = _fig_to_b64(fig1)
    st.markdown(
        f'<img src="data:image/png;base64,{b64_1}" '
        f'style="width:100%;border-radius:4px;margin-bottom:8px;" />',
        unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 2 — Catalytic bed outlet temperatures vs time
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="color:#00FF88;font-family:\'Courier New\',monospace;'
        'font-size:12px;font-weight:bold;border-left:3px solid #00FF88;'
        'padding-left:10px;margin:10px 0 6px 0;">2 — CATALYTIC BED OUTLET TEMPERATURES</div>',
        unsafe_allow_html=True)

    fig2, axes2 = _dyn_fig(1, 2, figsize=(18, 5))

    T_out_lits_t = hist.get('T_out_lits', [])

    # Graph A: T_out_bed(t) for all 4 beds on same axis
    ax = axes2[0]
    for i, (T_arr, col, lbl) in enumerate(zip(T_out_lits_t, COLORS_LITS, LABELS_LITS)):
        T_arr = np.array(T_arr)
        ax.plot(t, T_arr, color=col, lw=1.6, label=lbl)
    _dyn_style_ax(ax, title='Outlet temperature per catalytic bed (°C)',
                  xlabel='Time (s)', ylabel='Temperature (°C)')
    ax.legend(fontsize=8, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
              labelcolor=DYN_WHITE, loc='upper right', ncol=2)
    _add_step_line(ax, t_step)

    # Graph B: Absorption efficiencies JD02 / JD03 vs time
    ax = axes2[1]
    eff_jd02 = np.array(hist.get('eff_jd02', []))
    eff_jd03 = np.array(hist.get('eff_jd03', []))
    ax.plot(t, eff_jd02, color=DYN_CYAN,   lw=1.6, label='η JD02 inter')
    ax.plot(t, eff_jd03, color=DYN_ORANGE, lw=1.6, label='η JD03 final')
    ax.fill_between(t, eff_jd02, eff_jd03, alpha=0.08, color=DYN_WHITE)
    _dyn_style_ax(ax, title='Absorption efficiency JD02 / JD03 (%)',
                  xlabel='Time (s)', ylabel='η (%)')
    ax.legend(fontsize=8, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
              labelcolor=DYN_WHITE, loc='lower right')
    _add_step_line(ax, t_step)

    b64_2 = _fig_to_b64(fig2)
    st.markdown(
        f'<img src="data:image/png;base64,{b64_2}" '
        f'style="width:100%;border-radius:4px;margin-bottom:8px;" />',
        unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SECTION 3 — Converter spatial profiles (z-axis)
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="color:#00FF88;font-family:\'Courier New\',monospace;'
        'font-size:12px;font-weight:bold;border-left:3px solid #00FF88;'
        'padding-left:10px;margin:10px 0 6px 0;">3 — SPATIAL PROFILES — CONVERTER (z-axis)</div>',
        unsafe_allow_html=True)

    if profils:
        fig3, axes3 = _dyn_fig(1, 3, figsize=(18, 5))

        ax_T   = axes3[0]
        ax_tau = axes3[1]
        ax_r   = axes3[2]

        z_offset = 0.0
        lim_z    = []

        for i, (pl, col, lbl) in enumerate(zip(profils, COLORS_LITS, LABELS_LITS)):
            z   = np.array(pl.get('z',   []))
            T_C = np.array(pl.get('T_C', []))
            tau = np.array(pl.get('tau', []))
            r   = np.array(pl.get('r',   []))

            z_plot = z + z_offset

            ax_T.plot(z_plot, T_C, color=col, lw=1.8, label=lbl)
            ax_T.fill_between(z_plot, T_C, alpha=0.08, color=col)

            ax_tau.plot(z_plot, tau, color=col, lw=1.8, label=lbl)

            if len(r) > 0:
                r_norm = r / (np.max(np.abs(r)) + 1e-20)
                ax_r.plot(z_plot, r_norm, color=col, lw=1.8, label=lbl)

            if i < len(profils) - 1:
                z_offset_end = z_plot[-1]
                lim_z.append(z_offset_end)

            z_offset = z_plot[-1] + 0.3

        # Bed separation lines
        for ax_ in [ax_T, ax_tau, ax_r]:
            for zl in lim_z:
                ax_.axvline(x=zl + 0.15, color=DYN_GRAY, lw=1.0,
                            linestyle=':', alpha=0.6)

        _dyn_style_ax(ax_T,   title='T(z) profile per catalytic bed',
                      xlabel='Axial position (m)', ylabel='Temperature (°C)')
        _dyn_style_ax(ax_tau, title='τ(z) profile — Conversion rate (%)',
                      xlabel='Axial position (m)', ylabel='τ (%)')
        _dyn_style_ax(ax_r,   title='Normalized r(z) — Reaction rate',
                      xlabel='Axial position (m)', ylabel='Normalized r (-)')

        for ax_ in [ax_T, ax_tau, ax_r]:
            ax_.legend(fontsize=7.5, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
                       labelcolor=DYN_WHITE, loc='best', ncol=2)

        b64_3 = _fig_to_b64(fig3)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_3}" '
            f'style="width:100%;border-radius:4px;margin-bottom:8px;" />',
            unsafe_allow_html=True)
    else:
        st.info("Spatial profiles not available ('profils_lits' data missing).")

    # ══════════════════════════════════════════════════════════════
    # SECTION 4 — Furnace & boiler thermal profiles
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="color:#00FF88;font-family:\'Courier New\',monospace;'
        'font-size:12px;font-weight:bold;border-left:3px solid #00FF88;'
        'padding-left:10px;margin:10px 0 6px 0;">4 — THERMAL PROFILES — FURNACE & BOILER</div>',
        unsafe_allow_html=True)

    z_four_p = four.get('z_profil', [])
    T_four_p = four.get('T_profil', [])
    z_chaud_p    = chaud.get('z_profil', [])
    T_gaz_chaud  = chaud.get('T_gaz_profil', [])
    T_BFW_chaud  = chaud.get('T_BFW_profil', [])

    if z_four_p and z_chaud_p:
        fig4, axes4 = _dyn_fig(1, 2, figsize=(18, 5))

        # Furnace T profile
        ax = axes4[0]
        ax.plot(z_four_p, T_four_p, color=DYN_ORANGE, lw=2.0, label='Furnace gas T')
        ax.fill_between(z_four_p, T_four_p, alpha=0.12, color=DYN_ORANGE)
        # Flame temperature annotation
        idx_max = int(np.argmax(T_four_p))
        T_max   = float(np.max(T_four_p))
        ax.annotate(f'Flame T\n{T_max:.0f} °C',
                    xy=(z_four_p[idx_max], T_max),
                    xytext=(z_four_p[idx_max] + 0.05, T_max - 80),
                    color=DYN_YELLOW, fontsize=7.5, fontfamily='monospace',
                    arrowprops=dict(arrowstyle='->', color=DYN_YELLOW, lw=1.0))
        eta = four.get('eta_combustion', 0.0)
        ax.set_title(f'Furnace 401AF01 T profile — η comb. = {eta:.1f} %',
                     fontsize=9, fontweight='bold', fontfamily='monospace', color=DYN_GREEN)
        _dyn_style_ax(ax, xlabel='Normalized position (z/L)',
                      ylabel='Temperature (°C)')
        ax.legend(fontsize=8, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
                  labelcolor=DYN_WHITE, loc='upper right')

        # Boiler T profile (gas + BFW)
        ax = axes4[1]
        ax.plot(z_chaud_p, T_gaz_chaud, color=DYN_RED,  lw=2.0, label='Gas T (shell side)')
        ax.plot(z_chaud_p, T_BFW_chaud, color=DYN_CYAN, lw=2.0, label='BFW/Steam T')
        ax.fill_between(z_chaud_p, T_gaz_chaud, T_BFW_chaud,
                        alpha=0.08, color=DYN_WHITE, label='Pinch zone')
        eta_ch = chaud.get('eta_chaudiere', 0.0)
        ax.set_title(f'Boiler 401AV01 T profile — η = {eta_ch:.1f} %',
                     fontsize=9, fontweight='bold', fontfamily='monospace', color=DYN_GREEN)
        _dyn_style_ax(ax, xlabel='Normalized position (z/L)',
                      ylabel='Temperature (°C)')
        ax.legend(fontsize=8, facecolor=DYN_PANEL, edgecolor=DYN_GRID,
                  labelcolor=DYN_WHITE, loc='best')

        b64_4 = _fig_to_b64(fig4)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_4}" '
            f'style="width:100%;border-radius:4px;margin-bottom:8px;" />',
            unsafe_allow_html=True)
    else:
        st.info("Furnace/boiler thermal profiles not available.")

    # ══════════════════════════════════════════════════════════════
    # SECTION 5 — Conversion rate per bed (bar chart)
    # ══════════════════════════════════════════════════════════════
    st.markdown(
        '<div style="color:#00FF88;font-family:\'Courier New\',monospace;'
        'font-size:12px;font-weight:bold;border-left:3px solid #00FF88;'
        'padding-left:10px;margin:10px 0 6px 0;">5 — CONVERSION BALANCE & PRESSURE DROP PER BED</div>',
        unsafe_allow_html=True)

    tau_lits = conv.get('tau_lits', [])
    dp_lits  = conv.get('dp_lits',  [])

    if tau_lits and dp_lits:
        fig5, axes5 = _dyn_fig(1, 2, figsize=(14, 5))

        lits_labels = [f'Bed {i+1}' for i in range(len(tau_lits))]

        # τ bars per bed
        ax = axes5[0]
        bars = ax.bar(lits_labels, tau_lits, color=COLORS_LITS[:len(tau_lits)],
                      edgecolor=DYN_GRID, linewidth=0.8, width=0.55)
        for bar, val in zip(bars, tau_lits):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.2f}%', ha='center', va='bottom',
                    color=DYN_WHITE, fontsize=8.5, fontweight='bold',
                    fontfamily='monospace')
        _dyn_style_ax(ax, title='Cumulative conversion rate per bed (%)',
                      ylabel='Cumulative τ (%)', ylim=(0, 105))

        # ΔP bars per bed
        ax = axes5[1]
        dp_vals = [dp for dp in dp_lits]
        bars2 = ax.bar(lits_labels, dp_vals, color=COLORS_LITS[:len(dp_lits)],
                       edgecolor=DYN_GRID, linewidth=0.8, width=0.55)
        for bar, val in zip(bars2, dp_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom',
                    color=DYN_WHITE, fontsize=8.5, fontweight='bold',
                    fontfamily='monospace')
        _dyn_style_ax(ax, title='Pressure drop per catalytic bed (kPa)',
                      ylabel='ΔP (kPa)')

        b64_5 = _fig_to_b64(fig5)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_5}" '
            f'style="width:100%;border-radius:4px;margin-bottom:8px;" />',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# FIGURES — PROCESS VIEWS (unchanged logic)
# ══════════════════════════════════════════════════════════════════════

def draw_flowsheet(results, S_kgm, Air_nm3h, T_air, T_s, ratio_p,
                   turbo_res, dry_res, filter_res, tank_sys):
    fig, ax = plt.subplots(figsize=(32, 15))
    unisim_bg(fig, ax, xlim=(-2, 30), ylim=(-2.5, 11.5))
    draw_title_bar(ax, 'UNISIM-DESIGN SUPERVISION — SULFURIC ACID PLANT — GLOBAL PROCESS',
                   xlim=32, ylim=11.5)

    DY = -0.8

    st1 = tank_sys.sulfur_tank_1.state
    st2 = tank_sys.sulfur_tank_2.state
    sta = tank_sys.acid_tank.state

    BAC_W, BAC_H, CY_BAC = 0.90, 1.70, 5.0 + DY
    CX_S1, CX_S2 = -0.11, 1.34

    ax.add_patch(FancyBboxPatch((-0.71, CY_BAC - BAC_H / 2 - 0.55),
                               2.70, BAC_H + 1.30,
                               boxstyle="round,pad=0.06",
                               facecolor='#CCCCAA', edgecolor='#AA9900',
                               lw=1.5, alpha=0.5, zorder=1))
    ax.text(0.62, CY_BAC + BAC_H / 2 + 0.52,
            'SULFUR STORAGE', color='#AA6600', fontsize=7.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=8)

    alarm_s1 = st1.get('solidification_risk', False) or st1.get('overheat_risk', False)
    alarm_s2 = st2.get('solidification_risk', False) or st2.get('overheat_risk', False)

    draw_tank_unisim(ax, CX_S1, CY_BAC, BAC_W, BAC_H,
                     'TANK S1', st1.get('tag', '401AS01'),
                     st1['level_pct'], st1['T_C'], st1['density_kg_m3'],
                     '#DDCC00', alarm=alarm_s1,
                     extra=[(f"{st1['mass_kg']/1000:.2f}", "t")])
    draw_tank_unisim(ax, CX_S2, CY_BAC, BAC_W, BAC_H,
                     'TANK S2', st2.get('tag', '401AS02'),
                     st2['level_pct'], st2['T_C'], st2['density_kg_m3'],
                     '#DDCC00', alarm=alarm_s2,
                     extra=[(f"{st2['mass_kg']/1000:.2f}", "t")])

    CX_FIL = 2.0; CY_FIL = 2.0 + DY
    ax.add_patch(FancyBboxPatch((CX_FIL - 0.38, CY_FIL - 0.48), 0.76, 0.96,
                               boxstyle="round,pad=0.04",
                               facecolor=METAL_MID, edgecolor=METAL_BORDER, lw=1.8, zorder=3))
    for i in range(-2, 3):
        ax.plot([CX_FIL - 0.22, CX_FIL + 0.22], [CY_FIL + i * 0.10, CY_FIL - i * 0.10],
                color='#6688AA', lw=1.2, zorder=4)
    ax.text(CX_FIL, CY_FIL - 0.62, 'AIR FILTER\n401AS02',
            color='#111', fontsize=6.5, fontweight='bold',
            ha='center', va='top', fontfamily='monospace', zorder=5)
    data_box(ax, CX_FIL, CY_FIL - 1.5, [
        (f"{filter_res.get('T_out', 30):.2f}", "°C"),
        (f"{filter_res.get('delta_P_mmWC', 0):.2f}", "mmWC"),
    ], w=1.55)
    pipe(ax, CX_FIL - 1.50, CY_FIL, CX_FIL - 0.38, CY_FIL,
         color=C_AIR, lw=2.5, tag='F-AIR')

    CX_DT = 3.80; CY_DT = 2.0 + DY; DT_W = 0.90; DT_H = 1.55
    draw_absorption_tower_unisim(ax, CX_DT, CY_DT + 0.4, DT_W, DT_H,
                                 'DRYING\nTOWER', '401AD02')
    pipe(ax, CX_FIL + 0.38, CY_FIL, CX_DT - DT_W / 2, CY_DT,
         color=C_AIR, lw=2.5, tag='F110')
    pipe(ax, CX_DT, CY_DT + DT_H / 2 + 0.30, CX_DT, CY_DT + DT_H / 2,
         color=C_ACID_VIOLET, lw=2.0, tag='F119')
    data_box(ax, CX_DT + 0.55, CY_DT - 1.4, [
        (f"{dry_res.get('TG_out', T_air):.2f}", "°C gas"),
        (f"{dry_res.get('eff', 0):.2f}", "% eff"),
        (f"{dry_res.get('w_gNm3', 0):.4f}", "g/Nm³"),
    ], w=1.60)

    CX_TB = 6.80; CY_TB = 2.0 + DY; R_TB = 0.75
    draw_compressor_unisim(ax, CX_TB, CY_TB, R_TB,
                           title='TURBO BLOWER', tag='401AC01/02')
    pipe(ax, CX_DT + DT_W / 2, CY_DT, CX_TB - R_TB, CY_TB,
         color=C_AIR, lw=3.0, tag='F111')
    data_box(ax, CX_TB + R_TB + 0.10, CY_TB - 0.14, [
        (f"{turbo_res.get('N_rpm', 0):.2f}", "rpm"),
        (f"{turbo_res.get('delta_P_mmCE', 0):.2f}", "mmCE"),
        (f"{turbo_res.get('W_shaft_kW', 0):.2f}", "kW"),
        (f"{turbo_res.get('surge_margin_pct', 0):.2f}", "% surge"),
        (f"{turbo_res.get('T_out_C', 0):.2f}", "°C"),
        (f"{Air_nm3h:.2f}", "Nm³/h")
    ], w=1.65)

    CX_FOUR = 11.0; CY_FOUR = 5.2 + DY; FOUR_W = 3.5; FOUR_H = 1.55
    cylinder_3d(ax, CX_FOUR, CY_FOUR, FOUR_W, FOUR_H,
                label='SULFUR BURNER', tag='401AF01', horizontal=True)

    pipe(ax, CX_S2 + BAC_W / 2 + 1.65, CY_BAC, CX_FOUR - FOUR_W / 2, CY_FOUR + 0.35,
         color=C_SULFUR_ORG, lw=3.0, tag='F9')
    data_box(ax, 1.85, 4.50 + DY, [
        (f"{S_kgm:.2f}\n", "kg/min S"),
        (f"{T_s:.2f}", "°C"),
    ], w=1.60)
    ax.text(2.65, 5.25 + DY, 'LIQUID\nSULFUR', color="#E8E537",
            fontsize=7.5, fontweight='bold', ha='center', fontfamily='monospace', zorder=8)

    pipe(ax, CX_TB, CY_TB + R_TB + 0.80, CX_TB, CY_TB + R_TB,
         color=C_AIR, lw=3.0, arrow=False)
    pipe(ax, CX_TB, CY_TB + R_TB + 0.80, CX_FOUR - FOUR_W / 2, CY_FOUR,
         color=C_AIR, lw=3.0, tag='F112')

    data_box(ax, CX_FOUR - 0.70, CY_FOUR + FOUR_H / 2 + 0.12, [
        (f"{results['four']['SO2_pct']:.2f}", "% SO₂"),
        (f"{results['four']['O2_pct']:.2f}", "% O₂"),
        (f"{results['four']['T_out']:.2f}", "°C out"),
        (f"{results['four'].get('T_flamme', 0):.2f}", "°C fl."),
    ], w=1.65)

    CX_CHAUD = 16.0; CY_CHAUD = 5.2 + DY; CHAUD_W = 4.0; CHAUD_H = 1.80
    cylinder_3d(ax, CX_CHAUD, CY_CHAUD, CHAUD_W, CHAUD_H,
                label='WASTE HEAT BOILER', tag='401AV01', horizontal=True)

    pipe(ax, CX_FOUR + FOUR_W / 2, CY_FOUR, CX_CHAUD - CHAUD_W / 2, CY_CHAUD,
         color=C_GAZ_JAUNE, lw=4.0, tag='F-GAS')

    pipe(ax, CX_CHAUD, CY_CHAUD + CHAUD_H / 2, CX_CHAUD, CY_CHAUD + CHAUD_H / 2 + 0.8,
         color=C_STEAM_RED, lw=2.5, tag='F-STM')
    data_box(ax, CX_CHAUD + CHAUD_W / 2 - 2.10, CY_CHAUD - 2.20, [
        (f"{results['chaudiere']['T_out']:.2f}\n", "°C out"),
        (f"{results['chaudiere']['steam_flow']:.2f}\n", "t/h stm"),
        (f"{results['chaudiere']['power_mw']:.2f}\n", "MW"),
        (f"{results['chaudiere']['bypass_pct']:.2f}\n", "% bypass"),
    ], w=1.70)

    pipe(ax, CX_CHAUD, CY_CHAUD - CHAUD_H / 2 - 0.6, CX_CHAUD, CY_CHAUD - CHAUD_H / 2,
         color=C_WATER_BLUE, lw=2.0, tag='F-BFW')

    CX_LITS = 23.0; LIT_W = 2.2; LIT_H = 0.80; GAP = 1.55
    Y_LIT4 = 9.2 + DY; Y_LIT3 = Y_LIT4 - GAP
    Y_LIT2 = Y_LIT3 - GAP; Y_LIT1 = Y_LIT2 - GAP

    pipe(ax, CX_CHAUD + CHAUD_W / 2, CY_CHAUD, CX_LITS - LIT_W / 2, Y_LIT1,
         color=C_GAZ_JAUNE, lw=3.5, tag='F-B1IN')

    dp_lits = results.get('convertisseur', {}).get('dp_lits', [1.69, 1.75, 2.14, 1.90])
    for bed_idx, (Y_LIT, bed_num) in enumerate([(Y_LIT1, 1), (Y_LIT2, 2),
                                                 (Y_LIT3, 3), (Y_LIT4, 4)]):
        draw_bed_unisim(ax, CX_LITS, Y_LIT, LIT_W, LIT_H, bed_num,
                        results['convertisseur']['T_in_lits'][bed_idx],
                        results['convertisseur']['T_out_lits'][bed_idx],
                        results['convertisseur']['tau_lits'][bed_idx],
                        dp_kpa=dp_lits[bed_idx] if bed_idx < len(dp_lits) else 0.0)

    for ya, yb in [(Y_LIT1, Y_LIT2), (Y_LIT2, Y_LIT3)]:
        pipe(ax, CX_LITS, ya + LIT_H / 2, CX_LITS, yb - LIT_H / 2,
             color=C_GAZ_JAUNE, lw=2.5, arrow=True)

    CX_JD02 = 25.5; TOUR_W = 0.85; TOUR_H = 1.15
    pipe(ax, CX_LITS + LIT_W / 2, Y_LIT3, CX_JD02 - TOUR_W / 2, Y_LIT3,
         color=C_GAZ_JAUNE, lw=2.5, tag='F-B3OUT')
    draw_absorption_tower_unisim(ax, CX_JD02, Y_LIT3, TOUR_W, TOUR_H,
                                 'JD02\nINTER', '401AJ02')
    jd02 = results.get('jd02', {})
    data_box(ax, CX_JD02 + TOUR_W / 2 + 0.08, Y_LIT3 - 0.28, [
        (f"{jd02.get('T_gas_out', 0):.2f}", "°C gas"),
        (f"{jd02.get('eff_abs', 0):.2f}", "% η"),
        (f"{jd02.get('ppm_SO3_out', 0):.2f}", "ppm SO₃"),
    ], w=1.60)
    pipe(ax, CX_JD02, Y_LIT3 + TOUR_H / 2 + 0.30, CX_JD02, Y_LIT3 + TOUR_H / 2,
         color=C_ACID_VIOLET, lw=1.8, tag='F-AIN2')
    pipe(ax, CX_JD02, Y_LIT3 - TOUR_H / 2, CX_JD02, Y_LIT3 - TOUR_H / 2 - 0.30,
         color=C_ACID_VIOLET, lw=1.8, tag='F-AOUT2')

    pipe(ax, CX_JD02, Y_LIT3 + TOUR_H / 2, CX_JD02, Y_LIT4,
         color=C_AIR, lw=2.5, tag='F-B4IN')
    pipe(ax, CX_JD02, Y_LIT4, CX_LITS + LIT_W / 2, Y_LIT4, color=C_AIR, lw=2.5)

    CX_JD03 = 17.5
    pipe(ax, CX_LITS - LIT_W / 2, Y_LIT4, CX_JD03 + TOUR_W / 2, Y_LIT4,
         color=C_GAZ_JAUNE, lw=2.5, tag='F-B4OUT')
    draw_absorption_tower_unisim(ax, CX_JD03, Y_LIT4, TOUR_W, TOUR_H,
                                 'JD03\nFINAL', '401AJ03')
    jd03 = results.get('jd03', {})
    data_box(ax, CX_JD03 - TOUR_W / 2 - 1.70, Y_LIT4 - 0.28, [
        (f"{jd03.get('T_gas_out', 0):.2f}", "°C gas"),
        (f"{jd03.get('eff_abs', 0):.2f}", "% η"),
        (f"{jd03.get('ppm_SO3_out', 0):.2f}", "ppm SO₃"),
    ], w=1.60)
    pipe(ax, CX_JD03, Y_LIT4 + TOUR_H / 2 + 0.30, CX_JD03, Y_LIT4 + TOUR_H / 2,
         color=C_ACID_VIOLET, lw=1.8, tag='F-AIN3')
    pipe(ax, CX_JD03, Y_LIT4 - TOUR_H / 2, CX_JD03, Y_LIT4 - TOUR_H / 2 - 0.30,
         color=C_ACID_VIOLET, lw=1.8, tag='F-AOUT3')
    pipe(ax, CX_JD03, Y_LIT4 + TOUR_H / 2,
         CX_JD03, Y_LIT4 + TOUR_H / 2 + 0.55,
         color=C_AIR, lw=2.5, tag='F-STACK')
    ax.text(CX_JD03, Y_LIT4 + TOUR_H / 2 + 0.72,
            '→ STACK', color='#228822', fontsize=7.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=8)

    CX_BAC_A = 27.5; CY_BAC_A = 2.5 + DY; BAC_AW = 1.0; BAC_AH = 1.80
    alarm_a = sta.get('overheat_risk', False) or sta.get('dilution_risk', False)
    ax.add_patch(FancyBboxPatch((CX_BAC_A - BAC_AW / 2 - 0.20, CY_BAC_A - BAC_AH / 2 - 0.45),
                               BAC_AW + 0.40, BAC_AH + 1.10,
                               boxstyle="round,pad=0.06",
                               facecolor='#BBCCBB', edgecolor='#336633',
                               lw=1.5, alpha=0.5, zorder=1))
    ax.text(CX_BAC_A, CY_BAC_A + BAC_AH / 2 + 0.42,
            'ACID STORAGE', color='#226622', fontsize=7.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=8)
    draw_tank_unisim(ax, CX_BAC_A, CY_BAC_A, BAC_AW, BAC_AH,
                     'ACID TANK', sta.get('tag', '401AA01'),
                     sta['level_pct'], sta['T_C'], sta['density_kg_m3'],
                     '#9933CC', alarm=alarm_a,
                     extra=[(f"{tank_sys.total_acid_mass_t:.2f}", "t")])

    draw_pump_unisim(ax, CX_BAC_A, CY_BAC_A - BAC_AH / 2 - 0.55,
                     R=0.18, color='#DDDD33', label='PUMP', tag='401AP01')
    pipe(ax, CX_BAC_A, CY_BAC_A - BAC_AH / 2,
         CX_BAC_A, CY_BAC_A - BAC_AH / 2 - 0.37, color=C_ACID_VIOLET, lw=2.0)
    pipe(ax, CX_BAC_A, CY_BAC_A - BAC_AH / 2 - 0.73,
         CX_BAC_A, CY_BAC_A - BAC_AH / 2 - 1.05, color=C_ACID_VIOLET, lw=2.0)
    ax.text(CX_BAC_A, CY_BAC_A - BAC_AH / 2 - 1.12,
            '→ DISPATCH', color='#552288', fontsize=7, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=8)

    pipe(ax, CX_JD02, Y_LIT3 - TOUR_H / 2 - 0.30,
         CX_BAC_A - BAC_AW / 2, CY_BAC_A, color=C_ACID_VIOLET, lw=1.8, tag='F-RET2')
    pipe(ax, CX_JD03, Y_LIT4 - TOUR_H / 2 - 0.30,
         CX_BAC_A - BAC_AW / 2, CY_BAC_A + 0.20, color=C_ACID_VIOLET, lw=1.8, tag='F-RET3')

    draw_pump_unisim(ax, CX_DT + DT_W / 2 + 0.55, CY_DT + DT_H / 2 + 0.55,
                     R=0.15, color='#3366CC', label='', tag='401AP02')
    pipe(ax, CX_BAC_A - BAC_AW / 2, CY_BAC_A + 0.40,
         CX_DT + DT_W / 2 + 0.55, CY_DT + DT_H / 2 + 0.55,
         color=C_ACID_VIOLET, lw=1.8, tag='F-ACID-DT')
    pipe(ax, CX_DT + DT_W / 2 + 0.55, CY_DT + DT_H / 2 + 0.40,
         CX_DT, CY_DT + DT_H / 2 + 0.30, color=C_ACID_VIOLET, lw=1.8)

    leg_items = [
        (C_GAZ_JAUNE,   'Process gas / SO₂ / SO₃'),
        (C_AIR,         'Dry air'),
        (C_WATER_BLUE,  'Water / BFW'),
        (C_STEAM_RED,   'HP Steam'),
        (C_ACID_VIOLET, 'H₂SO₄ Acid'),
        (C_SULFUR_ORG,  'Liquid sulfur'),
        (C_SEA_WATER,   'Sea water'),
        (C_CONDENSAT,   'Condensate'),
    ]
    leg_x0 = 9.90; leg_y0 = -2.50
    ax.add_patch(FancyBboxPatch((leg_x0 - 0.10, leg_y0 - 0.10),
                               6.0, len(leg_items) * 0.30 + 0.30,
                               boxstyle="round,pad=0.06",
                               facecolor='#DDDDDD', edgecolor='#888',
                               lw=1.0, alpha=0.85, zorder=8))
    ax.text(leg_x0 + 2.80, leg_y0 + len(leg_items) * 0.30 + 0.10,
            'FLUID LEGEND', color='#111', fontsize=7.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=9)
    for i, (c, label) in enumerate(leg_items):
        yleg = leg_y0 + i * 0.30
        ax.plot([leg_x0 + 0.05, leg_x0 + 0.55], [yleg + 0.11, yleg + 0.11],
                color=c, lw=3.5, zorder=10)
        ax.text(leg_x0 + 0.65, yleg + 0.11, label, color='#111', fontsize=6.5,
                va='center', fontfamily='monospace', zorder=10)

    ax.text(28.50, -2.30, datetime.now().strftime('%d/%m/%Y  %H:%M:%S'),
            color='#333', fontsize=7, fontfamily='monospace', zorder=10)

    return fig


def render_page_equipements(filter_res, dry_res, turbo_res, Air_nm3h,
                            fouling_hours, Q_acid, T_acid, P_air_kPa):
    fig, ax = plt.subplots(figsize=(24, 10))
    unisim_bg(fig, ax, xlim=(0, 24), ylim=(0, 10))
    draw_title_bar(ax, 'AIR TREATMENT — FILTER / DRYING TOWER / TURBO BLOWER',
                   xlim=24, ylim=10)

    ax.add_patch(FancyBboxPatch((2.5, 4.0), 1.2, 1.6,
                               boxstyle="round,pad=0.06",
                               facecolor=METAL_MID, edgecolor=METAL_BORDER, lw=2.0, zorder=3))
    for i in range(-3, 4):
        ax.plot([2.70, 3.50], [4.80 + i * 0.14, 4.80 - i * 0.14],
                color='#6688AA', lw=1.2, zorder=4)
    ax.text(3.10, 3.82, 'AIR FILTER\n401AS02', color='#111', fontsize=8,
            fontweight='bold', ha='center', va='top', fontfamily='monospace', zorder=5)

    pipe(ax, 0.50, 4.80, 2.50, 4.80, color=C_AIR, lw=3.0, tag='F-AMB')
    data_box(ax, 0.10, 5.10, [
        ('AMBIENT AIR',),
        (f"T={30:.2f}", "°C"),
        (f"P={P_air_kPa:.2f}", "kPa"),
        (f"Q={Air_nm3h:.2f}", "Nm³/h"),
    ], w=2.10, color=FG_DATA)

    pipe(ax, 3.70, 4.80, 7.50, 4.80, color=C_AIR, lw=3.0, tag='F110')
    data_box(ax, 4.00, 5.10, [
        ('FILTERED AIR',),
        (f"T={filter_res.get('T_out', 30):.2f}", "°C"),
        (f"ΔP={filter_res.get('delta_P_mmWC', 0):.2f}", "mmWC"),
        (f"Foul={fouling_hours:.2f}", "h"),
    ], w=2.20)

    draw_absorption_tower_unisim(ax, 8.20, 4.80, 1.0, 2.0,
                                 'DRYING\nTOWER', '401AD02')

    pipe(ax, 8.20, 7.40, 8.20, 5.80, color=C_ACID_VIOLET, lw=2.0, tag='F119')
    data_box(ax, 7.70, 6.60, [
        ('H₂SO₄ IN',),
        (f"T={T_acid:.2f}", "°C"),
        (f"Q={Q_acid:.2f}", "m³/h"),
        ('w≈98.50', "%"),
    ], w=2.00)
    pipe(ax, 8.20, 3.80, 8.20, 2.80, color=C_ACID_VIOLET, lw=2.0, tag='F-ACIDOUT')
    data_box(ax, 7.70, 1.70, [
        ('H₂SO₄ OUT',),
        (f"T={dry_res.get('TL_out', T_acid):.2f}", "°C"),
        (f"Eff={dry_res.get('eff', 0):.2f}", "%"),
        (f"w={dry_res.get('w_H2SO4_in', 98.5):.2f}", "%"),
    ], w=2.10)

    pipe(ax, 8.70, 4.80, 14.00, 4.80, color=C_AIR, lw=3.0, tag='F111')
    data_box(ax, 9.50, 5.10, [
        ('DRY AIR',),
        (f"T={dry_res.get('TG_out', 30):.2f}", "°C"),
        (f"H₂O={dry_res.get('w_gNm3', 0):.4f}", "g/Nm³"),
    ], w=2.10)

    draw_compressor_unisim(ax, 15.0, 4.80, 1.0,
                           title='TURBO BLOWER', tag='401AC01/02')
    pipe(ax, 15.0, 5.80, 15.0, 7.20, color=C_AIR, lw=3.0, tag='F112')
    data_box(ax, 15.50, 6.60, [
        ('COMPRESSED AIR',),
        (f"T={turbo_res.get('T_out_C', 0):.2f}", "°C"),
        (f"ΔP={turbo_res.get('delta_P_mmCE', 0):.2f}", "mmCE"),
        (f"N={turbo_res.get('N_rpm', 0):.2f}", "rpm"),
        (f"W={turbo_res.get('W_shaft_kW', 0):.2f}", "kW"),
        (f"Surge={turbo_res.get('surge_margin_pct', 0):.2f}", "%"),
    ], w=2.10)
    pipe(ax, 15.0, 3.80, 15.0, 2.50, color='#CC8844', lw=1.8, tag='F-OIL')
    data_box(ax, 15.00, 2.00, [('LUBE OIL',), ('Lubrication circuit',)], w=2.00)

    return fig


def render_page_four_chaudiere(results, S_kgm, T_s, Air_nm3h, T_air, turbo_res):
    fig, ax = plt.subplots(figsize=(24, 10))
    unisim_bg(fig, ax, xlim=(0, 24), ylim=(0, 10))
    draw_title_bar(ax, 'COMBUSTION FURNACE & WASTE HEAT BOILER — 401AF01 / 401AV01',
                   xlim=24, ylim=10)

    cylinder_3d(ax, 7.0, 5.0, 4.5, 2.0, label='SULFUR BURNER', tag='401AF01', horizontal=True)

    pipe(ax, 1.50, 3.80, 4.75, 4.60, color=C_SULFUR_ORG, lw=3.5, tag='F9')
    data_box(ax, 0.10, 3.00, [
        ('LIQUID SULFUR',),
        (f"T={T_s:.2f}", "°C"),
        (f"Q={S_kgm:.2f}", "kg/min"),
        ('Purity≈99.90', "%"),
    ], w=2.20)

    pipe(ax, 7.0, 6.80, 7.0, 6.0, color=C_AIR, lw=3.0, tag='F112')
    data_box(ax, 7.50, 7.10, [
        ('DRY COMPRESSED AIR',),
        (f"T={turbo_res.get('T_out_C', 0):.2f}", "°C"),
        (f"Q={Air_nm3h:.2f}", "Nm³/h"),
        (f"ΔP={turbo_res.get('delta_P_mmCE', 0):.2f}", "mmCE"),
    ], w=2.20)

    pipe(ax, 9.25, 5.0, 12.20, 5.0, color=C_GAZ_JAUNE, lw=5.0, tag='F-GASB')
    data_box(ax, 10.20, 5.40, [
        (f"T={results['four']['T_out']:.2f}", "°C"),
        (f"SO₂={results['four']['SO2_pct']:.2f}", "%"),
        (f"O₂={results['four']['O2_pct']:.2f}", "%"),
        (f"T_fl={results['four'].get('T_flamme', 0):.2f}", "°C"),
    ], w=1.80)

    z2 = results['four'].get('z_profil', [])
    p2 = results['four'].get('T_profil', [])
    if len(z2) > 1 and len(p2) > 1:
        ax_ins = ax.inset_axes([0.28, 0.05, 0.14, 0.20])
        ax_ins.plot(z2, p2, color=C_STEAM_RED, lw=1.2)
        ax_ins.set_title('T(z) furnace [°C]', fontsize=5, color='#111')
        ax_ins.tick_params(labelsize=4)
        ax_ins.set_facecolor('#E8E8E8')

    cylinder_3d(ax, 16.0, 5.0, 5.0, 2.2, label='WASTE HEAT BOILER', tag='401AV01', horizontal=True)
    pipe(ax, 12.20, 5.0, 13.50, 5.0, color=C_GAZ_JAUNE, lw=5.0, arrow=False)

    pipe(ax, 16.0, 6.10, 16.0, 8.20, color=C_STEAM_RED, lw=3.0, tag='F-STM')
    data_box(ax, 16.50, 7.80, [
        ('HP STEAM',),
        ('T≈420.00', "°C"),
        ('P≈40.00', "bar"),
        (f"Q={results['chaudiere']['steam_flow']:.2f}", "t/h"),
        (f"P={results['chaudiere']['power_mw']:.2f}", "MW"),
    ], w=2.10)

    pipe(ax, 16.0, 2.20, 16.0, 3.90, color=C_WATER_BLUE, lw=2.5, tag='F-BFW')
    data_box(ax, 16.50, 1.90, [
        ('FEED WATER BFW',),
        ('T≈105.00', "°C"),
        ('P≈45.00', "bar"),
    ], w=2.10)

    pipe(ax, 18.50, 5.0, 21.50, 5.0, color=C_GAZ_JAUNE, lw=3.5, tag='F-CONV')
    data_box(ax, 19.50, 5.40, [
        (f"T={results['chaudiere']['T_out']:.2f}", "°C gas"),
        (f"Bypass={results['chaudiere']['bypass_pct']:.2f}", "%"),
    ], w=2.10)
    ax.text(21.70, 5.10, '→ CONV.', color='#335500', fontsize=8,
            fontweight='bold', ha='left', fontfamily='monospace', zorder=8)

    draw_valve_symbol(ax, 20.50, 5.0, r=0.15)

    return fig


def render_page_conv_absorption(results, T_in_lits_user):
    fig, ax = plt.subplots(figsize=(28, 14))
    unisim_bg(fig, ax, xlim=(0, 26), ylim=(-0.5, 12.5))
    draw_title_bar(ax, 'CONVERTER & ABSORPTION TOWERS — DETAILED FLOW SHEET',
                   xlim=26, ylim=12.5)

    conv = results['convertisseur']
    jd02 = results.get('jd02', {})
    jd03 = results.get('jd03', {})

    dp_lits = conv.get('dp_lits', [1.69, 1.75, 2.14, 1.90])

    CX_LITS = 13.0
    LIT_W   = 3.2
    LIT_H   = 0.80
    GAP     = 1.80

    Y_LIT1  = 2.0
    Y_LIT2  = Y_LIT1 + GAP
    Y_LIT3  = Y_LIT2 + GAP
    Y_LIT4  = Y_LIT3 + GAP

    CX_JD02 = 20.5
    CX_JD03 = 5.5
    TOUR_W  = 0.95
    TOUR_H  = 1.40

    T_ins  = conv['T_in_lits']
    T_outs = conv['T_out_lits']
    taus   = conv['tau_lits']

   

    data_box(ax, CX_LITS - LIT_W / 2 - 5.80, Y_LIT1 - 0.40, [
        ('INLET GAS',),
        (f"T={T_in_lits_user[0]:.2f}", "°C"),
        ('SO₂ + O₂ + N₂',),
    ], w=2.60)
    

    draw_bed_unisim(ax, CX_LITS, Y_LIT1, LIT_W, LIT_H, 1,
                    T_ins[0], T_outs[0], taus[0], dp_kpa=dp_lits[0])
    pipe(ax, CX_LITS, Y_LIT1 + LIT_H / 2, CX_LITS, Y_LIT2 - LIT_H / 2,
         color=C_GAZ_JAUNE, lw=3.0)
    draw_bed_unisim(ax, CX_LITS, Y_LIT2, LIT_W, LIT_H, 2,
                    T_ins[1], T_outs[1], taus[1], dp_kpa=dp_lits[1])
    pipe(ax, CX_LITS, Y_LIT2 + LIT_H / 2, CX_LITS, Y_LIT3 - LIT_H / 2,
         color=C_GAZ_JAUNE, lw=3.0)
    draw_bed_unisim(ax, CX_LITS, Y_LIT3, LIT_W, LIT_H, 3,
                    T_ins[2], T_outs[2], taus[2], dp_kpa=dp_lits[2])

    pipe(ax, CX_LITS + LIT_W / 2, Y_LIT3,
         CX_JD02 - TOUR_W / 2, Y_LIT3,
         color=C_GAZ_JAUNE, lw=3.0, tag='F-B3-JD02')

    draw_absorption_tower_unisim(ax, CX_JD02, Y_LIT3, TOUR_W, TOUR_H,
                                 'JD02\nINTER', '401AJ02')

    pipe(ax, CX_JD02, Y_LIT3 + TOUR_H / 2 + 0.50,
         CX_JD02, Y_LIT3 + TOUR_H / 2,
         color=C_ACID_VIOLET, lw=2.0, tag='F-AIN2')
    data_box(ax, CX_JD02 + 0.12, Y_LIT3 + TOUR_H / 2 + 0.52, [
        (f"H₂SO₄ IN  T={jd02.get('T_acid_in', 0):.2f}", "°C"),
        ('w ≈ 98.50', "%"),
    ], w=2.60)

    pipe(ax, CX_JD02, Y_LIT3 - TOUR_H / 2,
         CX_JD02, Y_LIT3 - TOUR_H / 2 - 0.45,
         color=C_ACID_VIOLET, lw=2.0, tag='F-AOUT2')

    data_box(ax, CX_JD02 + TOUR_W / 2 + 0.12, Y_LIT3 - 0.45, [
        (f"T_gas  = {jd02.get('T_gas_out',  0):.2f}", "°C"),
        (f"η      = {jd02.get('eff_abs',    0):.2f}", "%"),
        (f"SO₃    = {jd02.get('ppm_SO3_out', 0):.2f}", "ppm"),
        (f"T_acid = {jd02.get('T_acid_out', 0):.2f}", "°C"),
        (f"w      = {jd02.get('w_H2SO4_out', 0):.2f}", "% H₂SO₄"),
    ], w=2.20)

    Y_BYPASS = Y_LIT4 + 0.55
    pipe(ax, CX_JD02, Y_LIT3 + TOUR_H / 2,
         CX_JD02, Y_BYPASS,
         color=C_AIR, lw=2.5, tag='F-JD02-UP')
    pipe(ax, CX_JD02, Y_BYPASS,
         CX_LITS + LIT_W / 2, Y_BYPASS,
         color=C_AIR, lw=2.5)
    pipe(ax, CX_LITS + LIT_W / 2, Y_BYPASS,
         CX_LITS + LIT_W / 2, Y_LIT4 + LIT_H / 2,
         color=C_AIR, lw=2.5)
    data_box(ax, CX_LITS + LIT_W / 2 + 0.10, Y_BYPASS - 0.18, [
        (f"Desulfurized gas  T={jd02.get('T_gas_out', 0):.2f}", "°C"),
    ], w=2.80)

    draw_bed_unisim(ax, CX_LITS, Y_LIT4, LIT_W, LIT_H, 4,
                    T_ins[3], T_outs[3], taus[3], dp_kpa=dp_lits[3])

    pipe(ax, CX_LITS - LIT_W / 2, Y_LIT4,
         CX_JD03 + TOUR_W / 2, Y_LIT4,
         color=C_GAZ_JAUNE, lw=3.0, tag='F-B4-JD03')

    draw_absorption_tower_unisim(ax, CX_JD03, Y_LIT4, TOUR_W, TOUR_H,
                                 'JD03\nFINAL', '401AJ03')

    pipe(ax, CX_JD03, Y_LIT4 + TOUR_H / 2 + 0.50,
         CX_JD03, Y_LIT4 + TOUR_H / 2,
         color=C_ACID_VIOLET, lw=2.0, tag='F-AIN3')
    data_box(ax, CX_JD03 + 0.12, Y_LIT4 + TOUR_H / 2 + 0.52, [
        (f"H₂SO₄ IN  T={jd03.get('T_acid_in', 0):.2f}", "°C"),
        ('w ≈ 98.50', "%"),
    ], w=2.60)

    pipe(ax, CX_JD03, Y_LIT4 - TOUR_H / 2,
         CX_JD03, Y_LIT4 - TOUR_H / 2 - 0.45,
         color=C_ACID_VIOLET, lw=2.0, tag='F-AOUT3')

    data_box(ax, CX_JD03 - TOUR_W / 2 - 2.32, Y_LIT4 - 0.45, [
        (f"T_gas  = {jd03.get('T_gas_out',  0):.2f}", "°C"),
        (f"η      = {jd03.get('eff_abs',     0):.2f}", "%"),
        (f"SO₃    = {jd03.get('ppm_SO3_out', 0):.2f}", "ppm"),
        (f"T_acid = {jd03.get('T_acid_out', 0):.2f}", "°C"),
        (f"w      = {jd03.get('w_H2SO4_out', 0):.2f}", "% H₂SO₄"),
    ], w=2.20)

    pipe(ax, CX_JD03, Y_LIT4 + TOUR_H / 2,
         CX_JD03, Y_LIT4 + TOUR_H / 2 + 1.00,
         color=C_AIR, lw=2.5)
   

    return fig


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT CONFIG — UNISIM STYLE
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="UniSim-Design Supervision — Acid Plant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #B0B0B0 !important;
}
[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"]  { display: none !important; }
#MainMenu, footer, header         { visibility: hidden; }
.block-container { padding: 0.2rem 0.8rem 0 0.8rem !important; }
.unisim-titlebar {
    background: #3C3C3C; color: #FFFFFF;
    font-family: 'Courier New', monospace; font-size: 12px; font-weight: bold;
    padding: 4px 10px; display: flex; justify-content: space-between;
    align-items: center; border-bottom: 2px solid #222222; letter-spacing: 0.5px;
}
.unisim-winbtns { display: flex; gap: 4px; }
.unisim-winbtn  {
    width: 16px; height: 14px; border-radius: 2px; font-size: 9px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-weight: bold; color: white;
}
.btn-min { background: #666; } .btn-max { background: #666; } .btn-cls { background: #CC3333; }
.stTabs [data-baseweb="tab-list"] {
    background: #8A8A8A !important; border-bottom: 2px solid #555 !important;
    gap: 3px !important; padding: 3px 6px !important;
}
.stTabs [data-baseweb="tab"] {
    background: #707070 !important; color: #CCCCCC !important;
    font-family: 'Courier New', monospace !important; font-size: 11px !important;
    font-weight: bold !important; border: 1px solid #555 !important;
    border-radius: 2px !important; padding: 4px 14px !important;
    text-transform: uppercase; letter-spacing: 0.4px;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important; color: #111111 !important; border-color: #333 !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.ctrl-panel {
    background: #C8C8C8; border: 1px solid #888888; border-radius: 4px;
    padding: 8px 14px 5px 14px; margin-bottom: 6px;
}
.ctrl-title {
    color: #2a2a2a; font-weight: bold; font-size: 12px;
    font-family: 'Courier New', monospace; margin-bottom: 5px;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid #888; padding-bottom: 3px;
}
div[data-testid="stSlider"] label {
    color: #111111 !important; font-size: 11px !important;
    font-family: 'Courier New', monospace !important;
}
details summary {
    color: #111 !important; font-family: 'Courier New', monospace !important;
    font-size: 12px !important; font-weight: bold !important;
    background: #C0C0C0 !important; border: 1px solid #888 !important;
    border-radius: 3px !important; padding: 4px 10px !important;
}
details { background: #C8C8C8 !important; border: 1px solid #999 !important;
          border-radius: 4px !important; margin-bottom: 5px !important; }

[data-testid="stMarkdownContainer"] .dyn-section-title {
    color: #00FF88 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="unisim-titlebar">
  <span>Simulation Supervision  |  Sulfuric Acid Plant  |
        {datetime.now().strftime('%d/%m/%Y  %H:%M:%S')}</span>
  <div class="unisim-winbtns">
    <div class="unisim-winbtn btn-min">─</div>
    <div class="unisim-winbtn btn-max">□</div>
    <div class="unisim-winbtn btn-cls">✕</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── TankSystem initialization ────────────────────────────────────────
if 'tank_system' not in st.session_state:
    st.session_state.tank_system = TankSystem()


# ══════════════════════════════════════════════════════════════════════
# TABS — including tab6 DYNAMIC
# ══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "GLOBAL PROCESS",
    "AIR TREATMENT",
    "FURNACE & BOILER",
    "CONVERTER & EXCHANGERS",
    "CONVERTER & ABSORPTION",
    "DYNAMIC",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL PROCESS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    with st.expander("⚙ PROCESS PARAMETERS", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            S_kgm = st.slider("Sulfur (kg/min)", 500.0, 1500.0, DEFAULT_S, 10.0, key="g_s")
        with c2:
            T_s = st.slider("Sulfur T (°C)", 100.0, 160.0, DEFAULT_T_S, 1.0, key="g_ts")
        with c3:
            Air_nm3h = st.slider("Air (Nm³/h)", 250000.0, 500000.0, DEFAULT_AIR, 5000.0, key="g_air")
        with c4:
            T_air = st.slider("Air T (°C)", 80.0, 180.0, DEFAULT_T_AIR, 1.0, key="g_tair")
        with c5:
            ratio_p = st.slider("Primary air ratio", 0.3, 0.8, DEFAULT_RATIO, 0.01, key="g_ratio")

    with st.expander("SULFUR TANKS", expanded=False):
        bs1, bs2, bs3 = st.columns(3)
        with bs1:
            sulfur_in_1 = st.slider("Tank S1 delivery (m³/s)", 0.0, 0.05, 0.0, 0.001, format="%.3f", key="bs_in1")
        with bs2:
            sulfur_in_2 = st.slider("Tank S2 delivery (m³/s)", 0.0, 0.05, 0.0, 0.001, format="%.3f", key="bs_in2")
        with bs3:
            split_ratio = st.slider("Tank1/Tank2 → furnace split", 0.1, 0.9, 0.5, 0.05, key="bs_split")
        bs4, bs5, bs6 = st.columns(3)
        with bs4:
            sulfur_T_in = st.slider("Delivered sulfur T (°C)", 120.0, 150.0, 140.0, 1.0, key="bs_tin")
        with bs5:
            h_s1_init = st.slider("Initial level tank S1 (%)", 10.0, 100.0,
                                  st.session_state.tank_system.sulfur_tank_1.level_pct, 1.0, key="bs_h1")
        with bs6:
            h_s2_init = st.slider("Initial level tank S2 (%)", 10.0, 100.0,
                                  st.session_state.tank_system.sulfur_tank_2.level_pct, 1.0, key="bs_h2")

    with st.expander("ACID TANK", expanded=False):
        ba1, ba2, ba3, ba4 = st.columns(4)
        with ba1:
            acid_prod_m3s = st.slider("Acid production (m³/s)", 0.0, 0.10, 0.03, 0.001, format="%.3f", key="ba_prod")
        with ba2:
            acid_out_m3s = st.slider("Acid dispatch (m³/s)", 0.0, 0.10, 0.025, 0.001, format="%.3f", key="ba_out")
        with ba3:
            acid_T_in = st.slider("Incoming acid T (°C)", 30.0, 80.0, 55.0, 0.5, key="ba_tin")
        with ba4:
            dt_step = st.slider("Tank time step (s)", 10.0, 600.0, 60.0, 10.0, key="ba_dt")

    ts = st.session_state.tank_system
    if abs(ts.sulfur_tank_1.level_pct - h_s1_init) > 2.0:
        ts.sulfur_tank_1.h_B = h_s1_init / 100.0 * ts.sulfur_tank_1.h_max
    if abs(ts.sulfur_tank_2.level_pct - h_s2_init) > 2.0:
        ts.sulfur_tank_2.h_B = h_s2_init / 100.0 * ts.sulfur_tank_2.h_max

    ts.update_flows(
        S_kgmin=S_kgm, acid_prod_m3s=acid_prod_m3s, acid_out_m3s=acid_out_m3s,
        sulfur_in_1_m3s=sulfur_in_1, sulfur_in_2_m3s=sulfur_in_2,
        split_ratio=split_ratio, acid_T_in=acid_T_in, sulfur_T_in=sulfur_T_in,
    )
    ts.step(dt=float(dt_step))

    with st.spinner("Running dynamic calculation..."):
        results1 = simuler_complet(S_kgm, Air_nm3h, ratio_p, T_air, T_s)
        P_ATM = 101325.0
        af1 = AirFilter(); af1.Q_gas_Nm3h = Air_nm3h; af1.T_in = 30.0; af1.P_in = P_ATM
        try:
            fr1 = af1.compute(t_hours=0.0)
            if 'P_in_Pa' not in fr1: fr1['P_in_Pa'] = P_ATM
            Taf1 = fr1['T_out']
        except Exception:
            fr1 = {'delta_P_mmWC': 0.0, 'T_out': 30.0, 'P_in_Pa': P_ATM, 'P_out_Pa': P_ATM}
            Taf1 = 30.0
        dt1 = DryingTower(); dt1.Q_gas_Nm3h = Air_nm3h; dt1.Q_acid_m3h = 1245.0
        dt1.T_acid_in = 50.0; dt1.T_gas_in = Taf1
        try:
            dr1 = dt1.compute(); dr1['w_H2SO4_in'] = dt1.w_H2SO4_in * 100.0
        except Exception:
            dr1 = {'TG_out': Taf1, 'TL_out': 50.0, 'eff': 0.0, 'w_gNm3': 0.0, 'w_H2SO4_in': 98.6}
        tr1 = TurboBlowerTrain(); tr1.Q_gas_Nm3h = Air_nm3h; tr1.T_in = dr1.get('TG_out', Taf1)
        turbo1 = tr1.compute(load=1.0)

    if results1:
        tau_g = results1['convertisseur'].get('tau_final_pct', 0.0)
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Global converter τ", f"{tau_g:.2f} %")
        col_b.metric("Furnace outlet T", f"{results1['four']['T_out']:.1f} °C")
        col_c.metric("Steam produced", f"{results1['chaudiere']['steam_flow']:.1f} t/h")
        col_d.metric("Recovered power", f"{results1['chaudiere']['power_mw']:.2f} MW")

        fig1 = draw_flowsheet(results1, S_kgm, Air_nm3h, T_air, T_s, ratio_p,
                              turbo1, dr1, fr1, ts)
        b64_flow = _fig_to_b64(fig1, dpi=150)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_flow}" '
            f'style="width:100%;min-height:75vh;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
    else:
        st.error("Simulation error.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — AIR TREATMENT
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ PARAMETERS — AIR TREATMENT</div>',
                unsafe_allow_html=True)
    ca1, ca2, ca3, ca4, ca5 = st.columns(5)
    with ca1: Air_t2    = st.slider("Air flow (Nm³/h)",        250000.0, 500000.0, DEFAULT_AIR,  5000.0, key="at_air")
    with ca2: T_air_t2  = st.slider("Ambient air T (°C)",           10.0,     50.0,       30.0,     1.0, key="at_tair")
    with ca3: P_air_kPa = st.slider("Air pressure (kPa)",            95.0,    110.0,      101.3,     0.1, key="at_p")
    with ca4: Q_acid_t2 = st.slider("Acid flow (m³/h)",             800.0,  1800.0,     1245.0,    10.0, key="at_qa")
    with ca5: T_acid_t2 = st.slider("Acid inlet T (°C)",             40.0,     75.0,       50.0,     0.5, key="at_ta")
    st.markdown('</div>', unsafe_allow_html=True)

    P_ATM2 = P_air_kPa * 1000.0
    af2 = AirFilter(); af2.Q_gas_Nm3h = Air_t2; af2.T_in = T_air_t2; af2.P_in = P_ATM2
    try:
        fr2 = af2.compute(t_hours=0.0)
        if 'P_in_Pa' not in fr2: fr2['P_in_Pa'] = P_ATM2
        Taf2 = fr2['T_out']
    except Exception:
        fr2 = {'delta_P_mmWC': 0.0, 'T_out': T_air_t2, 'P_in_Pa': P_ATM2, 'P_out_Pa': P_ATM2}
        Taf2 = T_air_t2
    dt2 = DryingTower(); dt2.Q_gas_Nm3h = Air_t2; dt2.Q_acid_m3h = Q_acid_t2
    dt2.T_acid_in = T_acid_t2; dt2.T_gas_in = Taf2
    try:
        dr2 = dt2.compute(); dr2['w_H2SO4_in'] = dt2.w_H2SO4_in * 100.0
    except Exception:
        dr2 = {'TG_out': Taf2, 'TL_out': T_acid_t2, 'eff': 0.0, 'w_gNm3': 0.0, 'w_H2SO4_in': 98.6}
    tr2 = TurboBlowerTrain(); tr2.Q_gas_Nm3h = Air_t2; tr2.T_in = dr2.get('TG_out', Taf2)
    turbo2 = tr2.compute(load=1.0)

    fig2 = render_page_equipements(fr2, dr2, turbo2, Air_t2, 0.0, Q_acid_t2, T_acid_t2, P_air_kPa)
    b64_t2 = _fig_to_b64(fig2, dpi=140)
    st.markdown(
        f'<img src="data:image/png;base64,{b64_t2}" '
        f'style="width:100%;object-fit:contain;display:block;" />',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — COMBUSTION FURNACE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ PARAMETERS — FURNACE & BOILER</div>',
                unsafe_allow_html=True)
    cb1, cb2, cb3, cb4, cb5, cb6 = st.columns(6)
    with cb1: S_t3     = st.slider("Sulfur (kg/min)",       500.0,  1500.0, DEFAULT_S,           10.0, key="fb_s")
    with cb2: Ts_t3    = st.slider("Sulfur T (°C)",         100.0,   160.0, DEFAULT_T_S,           1.0, key="fb_ts")
    with cb3: Air_t3   = st.slider("Air flow (Nm³/h)",   250000.0, 500000.0, DEFAULT_AIR,       5000.0, key="fb_air")
    with cb4: Tair_t3  = st.slider("Air T (°C)",             80.0,   180.0, DEFAULT_T_AIR,         1.0, key="fb_tair")
    with cb5: ratio_t3 = st.slider("Secondary air ratio",    0.2,     0.7, 1.0 - DEFAULT_RATIO,  0.01, key="fb_ratio")
    with cb6: T_cible  = st.slider("Target conv. T (°C)",  380.0,   450.0, float(T_TARGET_CONV),  1.0, key="fb_tcible")
    st.markdown('</div>', unsafe_allow_html=True)

    ratio_p_t3 = 1.0 - ratio_t3
    with st.spinner("Computing furnace + boiler (dynamic)..."):
        res3 = simuler_complet(S_t3, Air_t3, ratio_p_t3, Tair_t3, Ts_t3)
    tr3 = TurboBlowerTrain(); tr3.Q_gas_Nm3h = Air_t3; tr3.T_in = 30.0
    turbo3 = tr3.compute(load=1.0)

    if res3:
        fig3 = render_page_four_chaudiere(res3, S_t3, Ts_t3, Air_t3, Tair_t3, turbo3)
        b64_t3 = _fig_to_b64(fig3, dpi=140)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_t3}" '
            f'style="width:100%;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
    else:
        st.error("Simulation error.")

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — CONVERTER & ABSORPTION
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ PARAMETERS — CONVERTER & ABSORPTION</div>',
                unsafe_allow_html=True)
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1: T_in_lit1 = st.slider("BED 1 inlet T (°C)", 380.0, 450.0, float(T_TARGET_CONV), 1.0, key="cv_t1")
    with cc2: T_in_lit2 = st.slider("BED 2 inlet T (°C)", 400.0, 480.0, 454.0, 1.0, key="cv_t2")
    with cc3: T_in_lit3 = st.slider("BED 3 inlet T (°C)", 400.0, 470.0, 449.0, 1.0, key="cv_t3")
    with cc4: T_in_lit4 = st.slider("BED 4 inlet T (°C)", 390.0, 450.0, 425.0, 1.0, key="cv_t4")
    with cc5: S_t4      = st.slider("Sulfur (kg/min)",     500.0, 1500.0, DEFAULT_S, 10.0, key="cv_s")
    st.markdown('</div>', unsafe_allow_html=True)

    T_in_lits_user = [T_in_lit1, T_in_lit2, T_in_lit3, T_in_lit4]
    with st.spinner("Computing converter + absorption (dynamic)..."):
        res4 = simuler_complet(S_t4, DEFAULT_AIR, DEFAULT_RATIO, DEFAULT_T_AIR, DEFAULT_T_S)

    if res4:
        res4['convertisseur']['T_in_lits'] = T_in_lits_user
        fig4 = render_page_conv_absorption(res4, T_in_lits_user)
        b64_t4 = _fig_to_b64(fig4, dpi=140)
        st.markdown(
            f'<img src="data:image/png;base64,{b64_t4}" '
            f'style="width:100%;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
    else:
        st.error("Simulation error.")

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — HEAT EXCHANGERS
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ PARAMETERS — HEAT EXCHANGERS</div>',
                unsafe_allow_html=True)
    hx_c1, hx_c2, hx_c3, hx_c4, hx_c5, hx_c6 = st.columns(6)
    with hx_c1: hx_S          = st.slider("Sulfur (kg/min)",          500.0,   1500.0, DEFAULT_S,   10.0, key="hx_s")
    with hx_c2: hx_air         = st.slider("Air (Nm³/h)",         250_000.0, 500_000.0, DEFAULT_AIR, 5_000.0, key="hx_air")
    with hx_c3: hx_T_sh_gas_in = st.slider("Gas → SH1B T (°C)",       580.0,    680.0,    627.0,    1.0, key="hx_sh_tin")
    with hx_c4: hx_T_hi_gas_in = st.slider("Gas → Hot IHX T (°C)",    480.0,    560.0,    516.0,    1.0, key="hx_hi_tin")
    with hx_c5: hx_T_ci_gas_in = st.slider("Gas → Cold IHX T (°C)",   420.0,    500.0,    460.0,    1.0, key="hx_ci_tin")
    with hx_c6: hx_T_bed4_out  = st.slider("Gas → HP4A T (°C)",       360.0,    450.0,    406.0,    1.0, key="hx_bed4_out")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Computing heat exchangers..."):
        _res_hx = simuler_complet(hx_S, hx_air, DEFAULT_RATIO, DEFAULT_T_AIR, DEFAULT_T_S)
        _comp_interpass = {'SO2': 0.05, 'SO3': 0.02, 'O2': 0.09, 'N2': 0.84}
        _comp_final     = {'SO2': 0.003, 'SO3': 0.05, 'O2': 0.10, 'N2': 0.847}
        _mdot_gas = hx_S / 60.0 * 2.0 * 5.5

        def _run_hx(cls, T_gas_in_C, P_gas=150_000.0, comp=None):
            hx = cls()
            hx.set_gas_inlet(
                T_C=T_gas_in_C, P_Pa=P_gas, mdot_kg_s=_mdot_gas,
                composition=(comp or _comp_interpass),
            )
            return hx.compute()

        r_sh   = _run_hx(HPSuperheater1B,  hx_T_sh_gas_in)
        r_hi   = _run_hx(HotInterpassHX,   hx_T_hi_gas_in)
        r_ci   = _run_hx(ColdInterpassHX,  hx_T_ci_gas_in)
        r_ec   = _run_hx(Economizer3B,     r_ci.get('T_gas_out_C', 306.0))
        r_hp4a = _run_hx(HP4AExchanger,    hx_T_bed4_out,                    comp=_comp_final)
        r_lp4a = _run_hx(LP4AExchanger,    r_hp4a.get('T_gas_out_C', 340.0), comp=_comp_final)
        r_e4c  = _run_hx(E4CExchanger,     r_lp4a.get('T_gas_out_C', 270.0), comp=_comp_final)
        r_e4a  = _run_hx(E4AExchanger,     r_e4c.get('T_gas_out_C', 200.0),  comp=_comp_final)

        hx_results = {
            'hp_superheater_1b': r_sh,
            'hot_interpass':     r_hi,
            'cold_interpass':    r_ci,
            'economizer_3b':     r_ec,
            'hp4a':  r_hp4a, 'lp4a': r_lp4a,
            'e4c':   r_e4c,  'e4a':  r_e4a,
        }
        _conv_res_hx = _res_hx.get('convertisseur', {}) if _res_hx else {}

        if 'T_in_lits' not in _conv_res_hx:
            _conv_res_hx['T_in_lits'] = [420.0, 454.0, 449.0, 425.0]
        if 'T_out_lits' not in _conv_res_hx:
            _conv_res_hx['T_out_lits'] = [580.0, 520.0, 480.0, 430.0]
        if 'dp_lits' not in _conv_res_hx:
            _conv_res_hx['dp_lits'] = [1.69, 1.75, 2.14, 1.90]

        fig5 = render_page_exchangers(hx_results, conv_results=_conv_res_hx)

    b64_t5 = _fig_to_b64(fig5, dpi=150)
    st.markdown(
        f'<img src="data:image/png;base64,{b64_t5}" '
        f'style="width:100%;min-height:75vh;object-fit:contain;display:block;" />',
        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TAB 6 — DYNAMIC
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ PARAMETERS — DYNAMIC SIMULATION</div>',
                unsafe_allow_html=True)
    dyn_c1, dyn_c2, dyn_c3, dyn_c4 = st.columns(4)
    with dyn_c1:
        dyn_S    = st.slider("Nominal sulfur (kg/min)", 500.0, 1500.0, DEFAULT_S,   10.0, key="dyn_s")
    with dyn_c2:
        dyn_air  = st.slider("Nominal air (Nm³/h)",    250000.0, 500000.0, DEFAULT_AIR, 5000.0, key="dyn_air")
    with dyn_c3:
        dyn_tair = st.slider("Air T (°C)",              80.0, 180.0, DEFAULT_T_AIR,  1.0, key="dyn_tair")
    with dyn_c4:
        dyn_ts   = st.slider("Sulfur T (°C)",          100.0, 160.0, DEFAULT_T_S,    1.0, key="dyn_ts")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Running dynamic simulation..."):
        res_dyn = simuler_complet(dyn_S, dyn_air, DEFAULT_RATIO, dyn_tair, dyn_ts)

    if res_dyn:
        render_page_dynamique(res_dyn)
    else:
        st.error("Error during dynamic simulation. Please check parameters.")
