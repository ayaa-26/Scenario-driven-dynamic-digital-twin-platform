import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import streamlit as st
from datetime import datetime
import io
import sys
import os
import base64

# ── Ajout du répertoire racine au path ────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from model.four import (
    DEFAULT_T_AIR_C  as DEFAULT_T_AIR,
    DEFAULT_T_SOUFRE_C as DEFAULT_T_S,
    DEFAULT_F_SOUFRE_KGMIN as DEFAULT_S,
    DEFAULT_F_AIR_NM3H as DEFAULT_AIR,
    DEFAULT_BYPASS_PCT,
)
DEFAULT_RATIO = 1.0 - DEFAULT_BYPASS_PCT / 100.0
from model.chaudiere import T_TARGET_CONV
from simulation.main_simulation import simuler_complet
from turbo_train import TurboBlowerTrain
from drying_tower import DryingTower
from air_filter import AirFilter

# ── Import échangeurs ─────────────────────────────────────────────────
from model.exchangers import (
    HPSuperheater1B, HotInterpassHX, ColdInterpassHX,
    Economizer3B,
    HP4AExchanger, LP4AExchanger, E4CExchanger, E4AExchanger,
)
from model.exchangers_page import render_page_exchangers

# ── Import bacs ───────────────────────────────────────────────────────
from model.bac import TankSystem, FlowIn, FlowOut

# ── Couleurs fluides ──────────────────────────────────────────────────
C_SOUFRE    = '#FFD700'
C_ACIDE_IN  = '#1a7a1a'
C_ACIDE_OUT = '#66cc44'
C_AIR       = '#f0f0f0'
C_GAZ       = '#E67E22'
C_VAP_HP    = '#FF2222'
C_VAP_BP    = '#FF9999'
C_EAU_MER   = '#003399'
C_EAU_DESSI = '#4499FF'
C_EAU_ALIM  = '#87CEEB'
C_HUILE     = '#D2B48C'
C_CONDENSAT = '#00FFFF'
C_BAC_S     = '#FFD700'   # bac soufre — jaune
C_BAC_A     = '#228B22'   # bac acide  — vert

VAL_BG  = '#111111'
VAL_FG  = '#00e040'
VAL_FG2 = '#ff9900'
VAL_FG3 = '#ffee00'

# =====================================================================
# HELPERS GRAPHIQUES
# =====================================================================

def draw_absorption_tower(ax, cx, cy, w, h, title, tag):
    from matplotlib.patches import Ellipse
    n = 30
    for i in range(n):
        frac = i / n
        gray = 0.30 + 0.35 * np.sin(np.pi * frac)
        ax.add_patch(plt.Rectangle(
            (cx - w/2, cy - h/2 + h*i/n), w, h/n,
            color=(gray, gray, gray), zorder=2))
    for yc, ec in [(cy - h/2, '#333'), (cy + h/2, '#555')]:
        ax.add_patch(Ellipse((cx, yc), width=w, height=w*0.3,
                             facecolor='#6a6a6a', edgecolor=ec, lw=1.2, zorder=3))
    ax.add_patch(FancyBboxPatch((cx - w/2, cy + h/2 - 0.04), w, 0.20,
                                boxstyle="round,pad=0.04", linewidth=1,
                                edgecolor='#555', facecolor='#7a7a7a', zorder=4))
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                               fill=False, edgecolor='#aaaaaa', lw=1.5, zorder=5))
    for y_pack in [cy - 0.15, cy + 0.25]:
        for sgn in [-1, 1]:
            ax.plot([cx - w/3, cx + w/3],
                    [y_pack - 0.13*sgn, y_pack + 0.13*sgn],
                    color='#aaaaee', lw=0.8, zorder=6, alpha=0.75)
    ax.text(cx, cy + h/2 + 0.28, title, color='white', fontsize=8,
            fontweight='bold', ha='center', va='bottom', zorder=7,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#2a5a7a')])
    ax.text(cx, cy - h/2 - 0.12, tag, color='#cccccc', fontsize=6,
            ha='center', va='top', zorder=7)


def draw_air_filter(ax, cx, cy, filter_res):
    w, h = 0.6, 0.8
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                               facecolor='#3a3a4a', edgecolor='#aaaaaa', lw=1.5, zorder=3))
    for i in range(-2, 3):
        ax.plot([cx - w/3, cx + w/3], [cy + i*0.12, cy - i*0.12],
                color='#88aacc', lw=1.2, zorder=4)
    ax.annotate("", xy=(cx - w/2, cy), xytext=(cx - w/2 - 0.25, cy),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_AIR, lw=1.5), zorder=8)
    ax.text(cx, cy - h/2 - 0.15, 'AIR\nFILTER', color='white', fontsize=6,
            fontweight='bold', ha='center', va='top', zorder=5,
            path_effects=[pe.withStroke(linewidth=1, foreground='#1a3a5a')])
    ax.text(cx, cy - h/2 - 0.32, '401AS02', color='#aaccee', fontsize=5,
            ha='center', va='top', zorder=5)


def draw_drying_tower(ax, cx, cy, w, h, title, tag, dry_res):
    from matplotlib.patches import Ellipse
    body_color = '#3a3a4a'
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                               facecolor=body_color, edgecolor='#aaaaaa', lw=1.8, zorder=3))
    ax.add_patch(Ellipse((cx, cy - h/2), width=w, height=w*0.35,
                         facecolor='#4a4a5a', edgecolor='#aaaaaa', lw=1.5, zorder=4))
    pack_y1 = cy - h/2 + h * 0.15
    pack_y2 = cy + h/2 - h * 0.30
    n_cross = 5
    for i in range(n_cross + 1):
        frac = i / n_cross
        yy = pack_y1 + (pack_y2 - pack_y1) * frac
        ax.plot([cx - w/2 + 0.03, cx + w/2 - 0.03], [yy, yy],
                color='#5a6a8a', lw=0.7, zorder=5, alpha=0.6)
    ax.plot([cx - w/2 + 0.03, cx + w/2 - 0.03], [pack_y1, pack_y2],
            color='#6a8aaa', lw=1.0, zorder=5)
    ax.plot([cx + w/2 - 0.03, cx - w/2 + 0.03], [pack_y1, pack_y2],
            color='#6a8aaa', lw=1.0, zorder=5)
    dist_y = cy + h/2 - h * 0.28
    ax.plot([cx - w/2 + 0.04, cx + w/2 - 0.04], [dist_y, dist_y],
            color=C_ACIDE_IN, lw=1.8, zorder=6)
    for xi in np.linspace(cx - w/2 + 0.06, cx + w/2 - 0.06, 5):
        ax.plot([xi, xi], [dist_y, dist_y - 0.08], color=C_ACIDE_IN, lw=1.2, zorder=6)
    gas_out_y = cy + h/2 - h * 0.05
    ax.plot([cx + w/2, cx + w/2 + 0.25], [gas_out_y, gas_out_y], color=C_AIR, lw=1.8, zorder=8)
    ax.text(cx, cy - h/2 - 0.38, title, color='white', fontsize=7,
            fontweight='bold', ha='center', va='top', zorder=9,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#1a3a5a')])
    ax.text(cx, cy - h/2 - 0.56, tag, color='#aaccee', fontsize=6,
            ha='center', va='top', zorder=9)


def draw_compressor_symbol(ax, cx, cy, R, title, tag, line_color='#aaaaaa'):
    outer = plt.Circle((cx, cy), R, facecolor='#2a2a3a', edgecolor=line_color,
                       linewidth=2.5, zorder=5)
    ax.add_patch(outer)
    r_inner = R * 0.42
    inner = plt.Circle((cx, cy), r_inner, facecolor='#1a1a2a', edgecolor=line_color,
                       linewidth=1.5, zorder=6)
    ax.add_patch(inner)
    tri_r = r_inner * 0.75
    angles_tri = [150, 270, 30]
    tri_pts = np.array([[cx + tri_r * np.cos(np.radians(a)),
                         cy + tri_r * np.sin(np.radians(a))]
                        for a in angles_tri])
    tri = plt.Polygon(tri_pts, closed=True, facecolor='#4a8aaa', edgecolor='white',
                      linewidth=1.2, zorder=7)
    ax.add_patch(tri)
    for angle_deg in range(0, 360, 60):
        a = np.radians(angle_deg)
        x0 = cx + r_inner * 0.55 * np.cos(a)
        y0 = cy + r_inner * 0.55 * np.sin(a)
        x1 = cx + r_inner * 0.95 * np.cos(a)
        y1 = cy + r_inner * 0.95 * np.sin(a)
        ax.plot([x0, x1], [y0, y1], color='#7ab0cc', lw=1.2, zorder=7)
    pipe_len = R * 0.85
    ax.annotate("", xy=(cx - R, cy), xytext=(cx - R - pipe_len, cy),
                arrowprops=dict(arrowstyle="->,head_width=0.18,head_length=0.12",
                                color=C_AIR, lw=2.0), zorder=8)
    ax.annotate("", xy=(cx, cy + R), xytext=(cx, cy + R + pipe_len),
                arrowprops=dict(arrowstyle="<-,head_width=0.18,head_length=0.12",
                                color=C_AIR, lw=2.0), zorder=8)
    ax.text(cx - R - pipe_len / 2, cy - 0.13, 'AIR SEC', color=C_AIR,
            fontsize=7, ha='center', fontweight='bold', zorder=9)
    ax.text(cx, cy - R - 0.18, title, color='white', fontsize=8,
            fontweight='bold', ha='center', va='top', zorder=9,
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#1a3a5a')])
    ax.text(cx, cy - R - 0.42, tag, color='#aaaacc', fontsize=7,
            ha='center', va='top', zorder=9)


# =====================================================================
# DESSIN DES BACS (helper)
# =====================================================================

def draw_tank(ax, cx, cy, w, h, label, tag, level_pct, T, density,
              color_fill, color_border, alarm=False, extra_lines=None):
    """
    Dessine un bac cylindrique vertical avec niveau, température, densité.
    alarm=True → bordure rouge clignotante.
    extra_lines : list of str affichées sous le bac.
    """
    from matplotlib.patches import Ellipse

    border_color = '#ff2222' if alarm else color_border
    border_lw    = 2.5 if alarm else 1.8

    # Corps vide
    ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                               facecolor='#222222', edgecolor=border_color,
                               lw=border_lw, zorder=3))

    # Remplissage selon niveau
    fill_h = max(0.0, min(1.0, level_pct / 100.0)) * h
    if fill_h > 0:
        ax.add_patch(plt.Rectangle((cx - w/2 + 0.03, cy - h/2 + 0.02),
                                   w - 0.06, fill_h - 0.02,
                                   facecolor=color_fill, alpha=0.72,
                                   edgecolor='none', zorder=4))

    # Ellipses haut/bas
    ew = w * 0.9
    for yc, fc, ec in [(cy - h/2, '#1a1a1a', '#444'),
                       (cy + h/2, '#3a3a3a', '#666')]:
        ax.add_patch(Ellipse((cx, yc), width=ew, height=w * 0.22,
                             facecolor=fc, edgecolor=ec, lw=1.2, zorder=5))

    # Ligne de niveau
    lvl_y = cy - h/2 + fill_h
    if 0 < fill_h < h:
        ax.plot([cx - w/2 + 0.04, cx + w/2 - 0.04], [lvl_y, lvl_y],
                color='white', lw=1.0, zorder=6, alpha=0.6)

    # Textes internes
    ax.text(cx, cy + 0.15, label, color='white', fontsize=7, fontweight='bold',
            ha='center', va='center', zorder=7,
            path_effects=[pe.withStroke(linewidth=1.2, foreground='black')])
    ax.text(cx, cy - 0.05, tag, color='#cccccc', fontsize=5.5,
            ha='center', va='center', zorder=7)
    ax.text(cx, cy - 0.25, f"{level_pct:.1f}%", color=VAL_FG3, fontsize=7,
            fontweight='bold', ha='center', va='center', zorder=7,
            path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])

    # Infos sous le bac
    info_y = cy - h/2 - 0.14
    ax.text(cx, info_y,
            f"T={T:.0f}°C  ρ={density:.0f}kg/m³",
            color=VAL_FG2, fontsize=5.5, ha='center', va='top', zorder=7)
    if extra_lines:
        for k, line in enumerate(extra_lines):
            ax.text(cx, info_y - 0.14*(k+1), line,
                    color='#ff8888' if alarm else '#aaaaaa',
                    fontsize=5, ha='center', va='top', zorder=7)


# =====================================================================
# FLOWSHEET GLOBAL (Page 1) — avec bacs
# =====================================================================

def draw_flowsheet(results, S_kgm, Air_nm3h, T_air, T_s, ratio_p,
                   turbo_res, dry_res, filter_res, tank_sys):

    BG = '#5a5a5a'
    fig, ax = plt.subplots(figsize=(30, 14))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-2, 26)
    ax.set_ylim(-2.5, 10.5)
    ax.axis('off')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    DY = -1.0

    def draw_path(x1, y1, x2, y2, color='#cc3333', lw=2.5, connectionstyle="arc3,rad=0"):
        arrow_props = dict(
            arrowstyle="->,head_width=0.4,head_length=0.5",
            color=color, linewidth=lw, mutation_scale=15, capstyle='round',
            connectionstyle=connectionstyle,
            path_effects=[pe.withStroke(linewidth=lw+1.5, foreground='#222222')])
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **arrow_props))

    def val_box(x, y, val, unit='', w=1.1, h=0.32, color=VAL_FG):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                              linewidth=1, edgecolor='#aaa', facecolor=VAL_BG)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, f"{val}", color=color,
                fontfamily='monospace', fontsize=11, fontweight='bold',
                va='center', ha='center',
                path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        if unit:
            ax.text(x + w - 0.08, y + h - 0.08, unit, color='#aaa',
                    fontsize=7, va='top', ha='right')

    def draw_cylinder(cx, cy, w, h, title, tag):
        from matplotlib.patches import Ellipse
        n = 40
        for i in range(n):
            frac = i / n
            gray = 0.35 + 0.4 * np.sin(np.pi * frac)
            ax.add_patch(plt.Rectangle(
                (cx - w/2, cy - h/2 + (h*i/n)), w, h/n,
                color=(gray, gray, gray), zorder=2))
        ax.plot([cx-w/2, cx+w/2], [cy-h/2, cy-h/2], color='#333', lw=2, zorder=3)
        ax.plot([cx-w/2, cx+w/2], [cy+h/2, cy+h/2], color='#333', lw=2, zorder=3)
        for xc in [cx-w/2, cx+w/2]:
            ax.add_patch(Ellipse((xc, cy), width=h*0.4, height=h,
                                 facecolor='#7a7a7a', edgecolor='#333', lw=1.5, zorder=4))
        ax.add_patch(plt.Rectangle((cx-w/2, cy-h/2), w, h,
                                   fill=False, edgecolor='#444', lw=1.5, zorder=5))
        ax.text(cx, cy+0.12, title, color='white', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax.text(cx, cy-0.15, tag, color='#cccccc', fontsize=7,
                ha='center', va='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])

    def draw_bed(cx, cy, w, h, bed_num, t_in, t_out, conv):
        from matplotlib.patches import Ellipse
        n = 25
        for i in range(n):
            frac = i / n
            gray = 0.45 + 0.3 * np.sin(np.pi * frac)
            ax.add_patch(plt.Rectangle(
                (cx-w/2, cy-h/2+(h*i/n)), w, h/n,
                color=(gray, gray, gray), zorder=2))
        for xc in [cx-w/2, cx+w/2]:
            ax.add_patch(Ellipse((xc, cy), width=h*0.35, height=h,
                                 facecolor='#6a6a6a', edgecolor='#333', lw=1, zorder=4))
        ax.add_patch(plt.Rectangle((cx-w/2, cy-h/2), w, h,
                                   fill=False, edgecolor='#444', lw=1, zorder=5))
        ax.text(cx, cy+0.1, f"BED {bed_num}", color='white', fontsize=8,
                fontweight='bold', ha='center', va='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        ax.text(cx-w/3, cy-0.08, f"{t_in:.0f}°C", color=VAL_FG3, fontsize=8,
                fontweight='bold', ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])
        ax.text(cx+w/3, cy-0.08, f"{t_out:.0f}°C",
                color=VAL_FG2 if t_out > 550 else VAL_FG, fontsize=8,
                fontweight='bold', ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])
        ax.text(cx, cy-0.28, f"{conv:.1f}%", color=VAL_FG, fontsize=7,
                fontweight='bold', ha='center', va='center',
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])

    # ── TITRE ────────────────────────────────────────────────────────
    ax.text(12, 10.1, 'SULFURIC ACID PLANT — PROCESS FLOW DIAGRAM',
            color='white', fontsize=14, fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='#2a5a7a')])

    # ================================================================
    # SECTION BACS — à gauche, x de -1.8 à 1.5
    # ================================================================
    st1 = tank_sys.sulfur_tank_1.state
    st2 = tank_sys.sulfur_tank_2.state
    sta = tank_sys.acid_tank.state

    BAC_W  = 0.85
    BAC_H  = 1.60
    CY_BAC = 4.8 + DY

    CX_S1 = -1.30
    CX_S2 =  0.05

    # Panneau fond bacs
    ax.add_patch(FancyBboxPatch((-1.85, CY_BAC - BAC_H/2 - 0.50),
                                2.55, BAC_H + 1.35,
                                boxstyle="round,pad=0.08",
                                linewidth=1.2, edgecolor='#aaaaaa',
                                facecolor='#2a2a2a', alpha=0.55, zorder=1))
    ax.text(-0.60, CY_BAC + BAC_H/2 + 0.58,
            'SULFUR STORAGE', color=C_BAC_S, fontsize=7,
            fontweight='bold', ha='center', va='center', zorder=8)

    # Bac soufre 1
    alarm_s1 = st1.get('solidification_risk', False) or st1.get('overheat_risk', False)
    draw_tank(ax, CX_S1, CY_BAC, BAC_W, BAC_H,
              'BAC S1', st1['tag'],
              st1['level_pct'], st1['T_C'], st1['density_kg_m3'],
              C_BAC_S, '#FFB300', alarm=alarm_s1,
              extra_lines=[f"m={st1['mass_kg']/1000:.0f}t",
                           '⚠ SOLID' if st1.get('solidification_risk') else ''])

    # Bac soufre 2
    alarm_s2 = st2.get('solidification_risk', False) or st2.get('overheat_risk', False)
    draw_tank(ax, CX_S2, CY_BAC, BAC_W, BAC_H,
              'BAC S2', st2['tag'],
              st2['level_pct'], st2['T_C'], st2['density_kg_m3'],
              C_BAC_S, '#FFB300', alarm=alarm_s2,
              extra_lines=[f"m={st2['mass_kg']/1000:.0f}t",
                           '⚠ SOLID' if st2.get('solidification_risk') else ''])

    # Autonomie soufre
    auto_h = tank_sys.autonomy_hours_sulfur
    auto_txt = f"Autono.: {auto_h:.1f} h" if auto_h < 1000 else "Autono.: ∞"
    ax.text(-0.60, CY_BAC - BAC_H/2 - 0.38, auto_txt,
            color=VAL_FG, fontsize=6, ha='center', va='center', zorder=8)
    ax.text(-0.60, CY_BAC - BAC_H/2 - 0.55,
            f"Stock total: {tank_sys.total_sulfur_mass_t:.0f} t",
            color=VAL_FG2, fontsize=6, ha='center', va='center', zorder=8)

    # Flèches soufre → four (S1 et S2 convergent vers le four)
    CX_FOUR = 6.5; CY_FOUR = 5.2 + DY
    draw_path(CX_S1 + BAC_W/2, CY_BAC,
              CX_FOUR - 1.6, CY_FOUR,
              color=C_SOUFRE, lw=2.5,
              connectionstyle="angle,angleA=0,angleB=45,rad=6")
    draw_path(CX_S2 + BAC_W/2, CY_BAC,
              CX_FOUR - 1.6, CY_FOUR,
              color=C_SOUFRE, lw=2.0,
              connectionstyle="angle,angleA=0,angleB=35,rad=6")

    val_box(1.2, 4.8+DY, f"{S_kgm:.0f}", 'kg/min', w=1.0, h=0.38, color=C_SOUFRE)
    val_box(1.2, 4.3+DY, f"{T_s:.0f}", '°C', w=0.8, h=0.38, color=C_SOUFRE)
    ax.text(1.7, 4.0+DY, 'SULFUR', color=C_SOUFRE, fontsize=8, fontweight='bold', ha='center')

    # ================================================================
    # FILTRE AIR
    # ================================================================
    CX_FILTER = 1.9; CY_FILTER = 1.8 + DY
    draw_air_filter(ax, CX_FILTER, CY_FILTER, filter_res)
    ax.annotate("", xy=(CX_FILTER - 0.3, CY_FILTER), xytext=(CX_FILTER - 0.8, CY_FILTER),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_AIR, lw=1.8), zorder=8)
    ax.text(CX_FILTER - 0.55, CY_FILTER + 0.4, 'AIR\nAMBIANT', color=C_AIR,
            fontsize=6, ha='center', zorder=9)
    val_box(CX_FILTER - 1.2, CY_FILTER + 0.1,
            f"{filter_res.get('T_out', 30):.0f}", '°C', w=0.7, h=0.24, color=VAL_FG3)

    # ================================================================
    # TOUR DE SÉCHAGE
    # ================================================================
    CX_DT = 3.5; CY_DT = 1.8 + DY; DT_W = 0.85; DT_H = 1.40
    draw_drying_tower(ax, cx=CX_DT, cy=CY_DT, w=DT_W, h=DT_H,
                      title='DRYING\nTOWER', tag='401AD02', dry_res=dry_res)
    draw_path(CX_FILTER + 0.3, CY_FILTER, CX_DT - DT_W/2, CY_DT, color=C_AIR, lw=2.5)

    # Flèche acide depuis bac acide → tour séchage
    # On place le bac acide plus bas à droite
    CX_BAC_A  = 22.0; CY_BAC_A = 1.5 + DY
    BAC_AW = 0.90; BAC_AH = 1.70

    # Acide tour séchage (annotation entrée acide drying tower)
    ax.annotate("", xy=(CX_DT, CY_DT + DT_H/2 - DT_H*0.28),
                xytext=(CX_DT, CY_DT + DT_H/2 + 0.20),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_ACIDE_IN, lw=1.4), zorder=9)
    ax.text(CX_DT + 0.55, CY_DT + DT_H/2 + 0.12, 'H₂SO₄\n(bac vert→)',
            color=C_ACIDE_IN, fontsize=5.5, ha='left', va='center', zorder=9)

    # ================================================================
    # TURBOSOUFFLANTE
    # ================================================================
    surge_color = '#ff4444' if turbo_res.get('is_surging', False) else VAL_FG
    CX_TB = 6.8; CY_TB = 1.8 + DY; R_TB = 0.72
    draw_compressor_symbol(ax, CX_TB, CY_TB, R_TB, title='MAIN COMPRESSORS',
                           tag='401AC01/02', line_color='#aaccee')
    val_box(CX_TB + R_TB + 0.08, CY_TB + 0.54,
            f"N:{turbo_res['N_rpm']:.0f}", 'rpm', w=1.30, h=0.28, color=VAL_FG2)
    val_box(CX_TB + R_TB + 0.08, CY_TB + 0.26,
            f"ΔP:{turbo_res['delta_P_mmCE']:.0f}", 'mmCE', w=1.30, h=0.28, color=VAL_FG3)
    val_box(CX_TB + R_TB + 0.08, CY_TB - 0.02,
            f"W:{turbo_res['W_shaft_kW']:.0f}", 'kW', w=1.30, h=0.28, color=VAL_FG2)
    val_box(CX_TB + R_TB + 0.08, CY_TB - 0.30,
            f"Surge:{turbo_res['surge_margin_pct']:.1f}", '%', w=1.45, h=0.28,
            color=surge_color)

    ax.annotate("", xy=(CX_TB, CY_TB - R_TB), xytext=(CX_TB, CY_TB - R_TB - 0.28),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_HUILE, lw=1.4), zorder=9)
    ax.text(CX_TB + 0.06, CY_TB - R_TB - 0.14, 'LUBE OIL',
            color=C_HUILE, fontsize=5, ha='left', va='center', zorder=9)

    gas_out_y_dt = CY_DT + DT_H/2 - DT_H * 0.05
    draw_path(CX_DT + DT_W/2 + 0.25, gas_out_y_dt,
              CX_DT + DT_W/2 + 0.25, CY_TB + 0.20, color=C_AIR, lw=3)
    draw_path(CX_DT + DT_W/2 + 0.25, CY_TB + 0.20,
              CX_TB - R_TB - 0.62, CY_TB, color=C_AIR, lw=3)
    val_box(CX_DT + DT_W/2 + 0.32, (gas_out_y_dt + CY_TB) / 2 - 0.10,
            f"{dry_res.get('TG_out', T_air):.0f}", '°C', w=0.85, h=0.28, color=VAL_FG3)

    pipe_len = R_TB * 0.85
    Y_sortie_TB = CY_TB + R_TB + pipe_len
    Y_four = 5.2 + DY
    draw_path(CX_TB, Y_sortie_TB, CX_TB, Y_four + 0.30, color=C_AIR, lw=3)
    draw_path(CX_TB, Y_four + 0.30, 6.5 - 1.6, Y_four, color=C_AIR, lw=3)
    val_box(CX_TB + 0.10, (Y_sortie_TB + Y_four) / 2,
            f"{turbo_res['T_out_C']:.0f}", '°C', w=0.85, h=0.28, color=VAL_FG3)
    ax.text(CX_TB + 0.15, (Y_sortie_TB + Y_four) / 2 - 0.28,
            'AIR COMPRIMÉ\nSEC', color=C_AIR, fontsize=6.5, ha='left')

    # ================================================================
    # FOUR
    # ================================================================
    draw_cylinder(6.5, 5.2+DY, 3.2, 1.4, 'SULFUR BURNER', '401-AF-01')
    val_box(5.2, 6.5+DY, f"O₂: {results['four']['O2_pct']:.2f}%", '', w=1.1, h=0.38)
    val_box(5.2, 6.0+DY, f"SO₂: {results['four']['SO2_pct']:.2f}%", '', w=1.2, h=0.38)

    # ================================================================
    # CHAUDIÈRE
    # ================================================================
    draw_cylinder(12.5, 5.2+DY, 3.6, 1.6, 'WASTE HEAT BOILER', '401-AV-01')
    draw_path(8.1, 5.2+DY, 10.7, 5.2+DY, color=C_GAZ, lw=4)
    val_box(9.0, 5.6+DY, f"{results['four']['T_out']:.0f}", '°C', w=0.8, h=0.38, color=VAL_FG3)
    val_box(11.0, 6.1+DY, f"T OUT: {results['chaudiere']['T_out']:.0f}°C", '',
            w=1.2, h=0.38,
            color=VAL_FG2 if results['chaudiere']['T_out'] > 400 else VAL_FG)
    val_box(12.0, 6.6+DY, f"{results['chaudiere']['steam_flow']:.1f}", 't/h',
            w=1.0, h=0.4, color=C_VAP_HP)
    val_box(13.0, 6.1+DY, f"{results['chaudiere']['power_mw']:.1f}", 'MW',
            w=0.9, h=0.38, color=C_VAP_HP)
    draw_path(10.7, 5.2+DY, 10.7, 7.0+DY, color=C_GAZ, lw=1.8,
              connectionstyle="angle,angleA=90,angleB=0,rad=0")
    draw_path(10.7, 7.0+DY, 14.3, 7.0+DY, color=C_GAZ, lw=1.8)
    draw_path(14.3, 7.0+DY, 14.3, 6.0+DY, color=C_GAZ, lw=1.8,
              connectionstyle="angle,angleA=0,angleB=90,rad=0")
    ax.text(12.5, 7.2+DY, f"BYPASS {results['chaudiere']['bypass_pct']:.1f}%",
            color='#ffaa44', fontsize=8, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2a2a2a', edgecolor='#ffaa44'))

    # ================================================================
    # CONVERTISSEUR + TOURS ABSORPTION
    # ================================================================
    CX_LITS = 17.0; LIT_W = 2.0; LIT_H = 0.75; GAP = 1.5
    Y_LIT4 = 8.5 + DY; Y_LIT3 = Y_LIT4 - GAP; Y_LIT2 = Y_LIT3 - GAP; Y_LIT1 = Y_LIT2 - GAP
    CX_JD02 = 20.5; CX_JD03 = 13.5; TOUR_W = 0.75; TOUR_H = 1.1

    draw_path(14.3, 5.2+DY, 14.3, Y_LIT1, color=C_GAZ, lw=3)
    draw_path(14.3, Y_LIT1, CX_LITS - LIT_W/2, Y_LIT1, color=C_GAZ, lw=3)
    val_box(14.7, Y_LIT1 + 0.5, f"{T_TARGET_CONV:.0f}", '°C', w=0.8, h=0.32, color=VAL_FG3)
    ax.text(14.75, Y_LIT1 + 0.25, 'GAS IN', color='#ffaa66', fontsize=7, ha='center')

    draw_bed(CX_LITS, Y_LIT1, LIT_W, LIT_H, 1,
             results['convertisseur']['T_in_lits'][0],
             results['convertisseur']['T_out_lits'][0],
             results['convertisseur']['tau_lits'][0])
    draw_path(CX_LITS, Y_LIT1 + LIT_H/2 + 0.05,
              CX_LITS, Y_LIT2 - LIT_H/2 - 0.05, color=C_GAZ, lw=2.2)
    draw_bed(CX_LITS, Y_LIT2, LIT_W, LIT_H, 2,
             results['convertisseur']['T_in_lits'][1],
             results['convertisseur']['T_out_lits'][1],
             results['convertisseur']['tau_lits'][1])
    draw_path(CX_LITS, Y_LIT2 + LIT_H/2 + 0.05,
              CX_LITS, Y_LIT3 - LIT_H/2 - 0.05, color=C_GAZ, lw=2.2)
    draw_bed(CX_LITS, Y_LIT3, LIT_W, LIT_H, 3,
             results['convertisseur']['T_in_lits'][2],
             results['convertisseur']['T_out_lits'][2],
             results['convertisseur']['tau_lits'][2])
    draw_path(CX_LITS + LIT_W/2, Y_LIT3,
              CX_JD02 - TOUR_W/2, Y_LIT3, color=C_GAZ, lw=2.5)
    ax.text((CX_LITS + LIT_W/2 + CX_JD02 - TOUR_W/2) / 2, Y_LIT3 + 0.16,
            'BED3→JD02', color='#ffaa44', fontsize=7, ha='center', fontweight='bold')
    draw_absorption_tower(ax, cx=CX_JD02, cy=Y_LIT3, w=TOUR_W, h=TOUR_H,
                          title='JD02', tag='InterTower.')
    jd02 = results.get('jd02', {})
    val_box(CX_JD02 + TOUR_W/2 + 0.08, Y_LIT3 + 0.22,
            f"T:{jd02.get('T_gas_out',0):.0f}°C", '', w=0.85, h=0.26, color=VAL_FG)
    val_box(CX_JD02 + TOUR_W/2 + 0.08, Y_LIT3 - 0.06,
            f"η:{jd02.get('eff_abs',0):.1f}%", '', w=0.85, h=0.26, color=VAL_FG2)
    val_box(CX_JD02 + TOUR_W/2 + 0.08, Y_LIT3 - 0.34,
            f"{jd02.get('ppm_SO3_out',0):.0f}ppm", '', w=0.85, h=0.26, color=VAL_FG3)

    # Acide entrant/sortant JD02
    ax.annotate("", xy=(CX_JD02, Y_LIT3 + TOUR_H/2),
                xytext=(CX_JD02, Y_LIT3 + TOUR_H/2 + 0.22),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_ACIDE_IN, lw=1.4), zorder=9)
    ax.annotate("", xy=(CX_JD02, Y_LIT3 - TOUR_H/2 - 0.18),
                xytext=(CX_JD02, Y_LIT3 - TOUR_H/2),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_ACIDE_OUT, lw=1.4), zorder=9)

    # Flèche retour bac acide ← JD02 (acide produit)
    draw_path(CX_JD02, Y_LIT3 - TOUR_H/2 - 0.20,
              CX_BAC_A, CY_BAC_A + BAC_AH/2,
              color=C_ACIDE_OUT, lw=1.8,
              connectionstyle="arc3,rad=-0.15")
    ax.text((CX_JD02 + CX_BAC_A)/2 + 0.5, Y_LIT3 - TOUR_H/2 - 0.55,
            'Acide produit → bac vert', color=C_ACIDE_OUT, fontsize=5.5,
            ha='center', va='top', zorder=9)

    draw_path(CX_JD02, Y_LIT3 + TOUR_H/2 + 0.05, CX_JD02, Y_LIT4, color=C_AIR, lw=2.5)
    draw_path(CX_JD02, Y_LIT4, CX_LITS + LIT_W/2, Y_LIT4, color=C_AIR, lw=2.5)
    ax.text((CX_JD02 + CX_LITS + LIT_W/2) / 2, Y_LIT4 + 0.15,
            'JD02→BED4', color='#88aaff', fontsize=7, ha='center', fontweight='bold')
    draw_bed(CX_LITS, Y_LIT4, LIT_W, LIT_H, 4,
             results['convertisseur']['T_in_lits'][3],
             results['convertisseur']['T_out_lits'][3],
             results['convertisseur']['tau_lits'][3])
    draw_path(CX_LITS - LIT_W/2, Y_LIT4,
              CX_JD03 + TOUR_W/2, Y_LIT4, color=C_GAZ, lw=2.5)
    ax.text((CX_LITS - LIT_W/2 + CX_JD03 + TOUR_W/2) / 2, Y_LIT4 + 0.16,
            'BED4→JD03', color='#ffaa44', fontsize=7, ha='center', fontweight='bold')
    draw_absorption_tower(ax, cx=CX_JD03, cy=Y_LIT4, w=TOUR_W, h=TOUR_H,
                          title='JD03', tag='Final Tower')
    jd03 = results.get('jd03', {})
    val_box(CX_JD03 - TOUR_W/2 - 0.93, Y_LIT4 + 0.22,
            f"T:{jd03.get('T_gas_out',0):.0f}°C", '', w=0.85, h=0.26, color=VAL_FG)
    val_box(CX_JD03 - TOUR_W/2 - 0.93, Y_LIT4 - 0.06,
            f"η:{jd03.get('eff_abs',0):.1f}%", '', w=0.85, h=0.26, color=VAL_FG2)
    val_box(CX_JD03 - TOUR_W/2 - 0.93, Y_LIT4 - 0.34,
            f"{jd03.get('ppm_SO3_out',0):.0f}ppm", '', w=0.85, h=0.26, color=VAL_FG3)

    ax.annotate("", xy=(CX_JD03, Y_LIT4 + TOUR_H/2),
                xytext=(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.22),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_ACIDE_IN, lw=1.4), zorder=9)
    ax.annotate("", xy=(CX_JD03, Y_LIT4 - TOUR_H/2 - 0.18),
                xytext=(CX_JD03, Y_LIT4 - TOUR_H/2),
                arrowprops=dict(arrowstyle="->,head_width=0.09,head_length=0.08",
                                color=C_ACIDE_OUT, lw=1.4), zorder=9)

    # Flèche retour bac acide ← JD03 (acide produit)
    draw_path(CX_JD03, Y_LIT4 - TOUR_H/2 - 0.20,
              CX_BAC_A, CY_BAC_A + BAC_AH/2 + 0.10,
              color=C_ACIDE_OUT, lw=1.8,
              connectionstyle="arc3,rad=0.12")

    draw_path(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.05,
              CX_JD03, Y_LIT4 + TOUR_H/2 + 0.7, color=C_AIR, lw=2.5)
    ax.text(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.85, 'GAZ TRAITÉ',
            color='#88aaff', fontsize=7, ha='center', fontweight='bold')

    # ================================================================
    # BAC ACIDE — à droite du flowsheet
    # ================================================================
    alarm_a = sta.get('overheat_risk', False) or sta.get('dilution_risk', False)

    # Panneau fond bac acide
    ax.add_patch(FancyBboxPatch((CX_BAC_A - BAC_AW/2 - 0.20, CY_BAC_A - BAC_AH/2 - 0.55),
                                BAC_AW + 0.40, BAC_AH + 1.40,
                                boxstyle="round,pad=0.08",
                                linewidth=1.2, edgecolor='#44aa44',
                                facecolor='#1a2a1a', alpha=0.60, zorder=1))
    ax.text(CX_BAC_A, CY_BAC_A + BAC_AH/2 + 0.58,
            'ACID STORAGE', color=C_BAC_A, fontsize=7,
            fontweight='bold', ha='center', va='center', zorder=8)

    draw_tank(ax, CX_BAC_A, CY_BAC_A, BAC_AW, BAC_AH,
              'BAC ACIDE', sta['tag'],
              sta['level_pct'], sta['T_C'], sta['density_kg_m3'],
              C_BAC_A, '#228B22', alarm=alarm_a,)
            

    ax.text(CX_BAC_A, CY_BAC_A - BAC_AH/2 + 0.5,
            f"Stock: {tank_sys.total_acid_mass_t:.0f} t",
            color=VAL_FG2, fontsize=6, ha='center', va='center', zorder=8)

    # Flèche expédition (sortie bas bac acide)
    ax.annotate("", xy=(CX_BAC_A, CY_BAC_A - BAC_AH/2 - 0.65),
                xytext=(CX_BAC_A, CY_BAC_A - BAC_AH/2),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_ACIDE_OUT, lw=1.8), zorder=9)
    ax.text(CX_BAC_A, CY_BAC_A - BAC_AH/2 - 0.70,
            'Expédition', color=C_ACIDE_OUT, fontsize=6,
            ha='center', va='top', zorder=9)

    # Flèche bac acide → tour séchage (entrée acide sur drying tower)
    draw_path(CX_BAC_A - BAC_AW/2, CY_BAC_A,
              CX_DT + DT_W/2 + 0.15, CY_DT + DT_H/2 + 0.18,
              color=C_ACIDE_IN, lw=1.8,
              connectionstyle="arc3,rad=0.22")
    ax.text((CX_BAC_A + CX_DT)/2, CY_BAC_A + 0.35,
            'H₂SO₄ → séchage', color=C_ACIDE_IN, fontsize=5.5,
            ha='center', va='bottom', zorder=9)

    # ================================================================
    # LÉGENDE
    # ================================================================
    legend_fluid = [
        (C_SOUFRE,    'Soufre'),
        (C_ACIDE_IN,  'Acide H₂SO₄ entrée'),
        (C_ACIDE_OUT, 'Acide H₂SO₄ sortie'),
        (C_AIR,       'Air'),
        (C_GAZ,       'Gaz process'),
        (C_VAP_HP,    'Vapeur HP'),
        (C_EAU_ALIM,  'Eau alimentaire (BFW)'),
        (C_HUILE,     'Huile lubrification'),
        (C_CONDENSAT, 'Condensat'),
        (C_BAC_S,     'Bac soufre (jaune)'),
        (C_BAC_A,     'Bac acide  (vert)'),
    ]
    leg_x0 = 25.5; leg_y0 = -2.20
    ax.text(leg_x0 + 0.22, leg_y0 + len(legend_fluid)*0.27 + 0.10,
            'LÉGENDE FLUIDES', color='white', fontsize=7, fontweight='bold',
            ha='left', va='bottom', zorder=10)
    for i, (c, label) in enumerate(legend_fluid):
        yleg = leg_y0 + i * 0.27
        ax.plot([leg_x0, leg_x0 + 0.40], [yleg, yleg], color=c, lw=2.5, zorder=10)
        ax.text(leg_x0 + 0.50, yleg, label, color='#dddddd', fontsize=6.5,
                va='center', zorder=10)

    ax.text(-1.8, -2.3, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), color='#888', fontsize=7)
    return fig


# =====================================================================
# PAGE 2 — AIR TREATMENT
# =====================================================================

def render_page_equipements(filter_res, dry_res, turbo_res, Air_nm3h,
                            fouling_hours, Q_acid, T_acid, P_air_kPa):
    BG = '#5a5a5a'
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    def arrow(x1, y1, x2, y2, color, lw=2.0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.20",
                                   color=color, lw=lw, connectionstyle="arc3,rad=0.0"), zorder=8)

    def flow_box(x, y, txt, color=VAL_FG, fs=7.5):
        ax.text(x, y, txt, color=color, fontsize=fs, fontweight='bold',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=VAL_BG,
                          edgecolor=color, lw=1.4))

    ax.text(11, 8.6, 'AIR TREATMENT SECTION — Detailed Streams',
            color='white', fontsize=13, fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='#2a5a7a')])

    draw_air_filter(ax, 3.5, 4.5, filter_res)
    ax.text(3.5, 5.0, 'AIR FILTER\n401AS02', color='white', fontsize=9,
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#1a3a5a')])
    arrow(1.2, 4.5, 3.2, 4.5, C_AIR, lw=2.5)
    flow_box(1.9, 5.0,
             f"AIR AMBIANT\nT = 30°C\nP = {P_air_kPa:.1f} kPa\nQ = {Air_nm3h:.0f} Nm³/h",
             C_AIR)
    arrow(3.8, 4.5, 7.5, 4.5, C_AIR, lw=2.5)
    flow_box(5.0, 5.0,
             f"AIR FILTRÉ\nT = {filter_res.get('T_out',30):.1f}°C\n"
             f"ΔP = {filter_res.get('delta_P_mmWC',0):.1f} mmWC\n"
             f"Colmatage = {fouling_hours:.0f} h",
             "#f4f6f7")

    draw_drying_tower(ax, cx=8.0, cy=4.5, w=1.0, h=1.8,
                      title='DRYING\nTOWER', tag='401AD02', dry_res=dry_res)
    ax.text(8.0, 7.1, 'DRYING TOWER\n401AD02', color='white', fontsize=9,
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#1a3a5a')])
    arrow(8.0, 6.5, 8.0, 5.4, C_ACIDE_IN, lw=2.0)
    flow_box(8.0, 6.5,
             f"H₂SO₄ IN\nT = {T_acid:.1f}°C\nQ = {Q_acid:.0f} m³/h\nw ≈ 98.5 %",
             C_ACIDE_IN)
    arrow(8.0, 3.6, 8.0, 2.7, C_ACIDE_OUT, lw=2.0)
    flow_box(8.0, 2.4,
             f"H₂SO₄ OUT\nT = {dry_res.get('TL_out', T_acid):.1f}°C\n"
             f"Eff = {dry_res.get('eff',0):.1f} %\nw = {dry_res.get('w_H2SO4_in',98.5):.1f} %",
             C_ACIDE_OUT)
    gas_y = 4.5 + 0.9 - 1.8 * 0.05
    arrow(8.5, 4.5, 13.0, 4.5, C_AIR, lw=2.5)
    flow_box(9.3, gas_y - 0.25,
             f"AIR SEC\nT = {dry_res.get('TG_out', 30):.1f}°C\n"
             f"H₂O = {dry_res.get('w_gNm3',0):.3f} g/Nm³",
             C_AIR)

    draw_compressor_symbol(ax, 14.5, 4.5, 0.9,
                           title='TURBO-BLOWER\n401AC01/02', tag='401AC01/02',
                           line_color='#aaccee')
    flow_box(14.4, 6.6,
             f"AIR COMPRIMÉ\nT = {turbo_res.get('T_out_C',0):.1f}°C\n"
             f"ΔP = {turbo_res.get('delta_P_mmCE',0):.0f} mmCE\n"
             f"N = {turbo_res.get('N_rpm',0):.0f} rpm\n"
             f"W = {turbo_res.get('W_shaft_kW',0):.0f} kW\n"
             f"Surge margin = {turbo_res.get('surge_margin_pct',0):.1f} %",
             C_AIR)
    arrow(14.5, 3.6, 14.5, 2.7, C_HUILE, lw=1.8)
    flow_box(14.5, 2.5, "LUBE OIL\nCircuit lubrification", C_HUILE)
    return fig


# =====================================================================
# PAGE 3 — FOUR + CHAUDIÈRE
# =====================================================================

def render_page_four_chaudiere(results, S_kgm, T_s, Air_nm3h, T_air, turbo_res):
    BG = '#5a5a5a'
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    def arrow(x1, y1, x2, y2, color, lw=2.0, cs="arc3,rad=0.0"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.20",
                                   color=color, lw=lw, connectionstyle=cs), zorder=8)

    def flow_box(x, y, txt, color=VAL_FG, fs=7.5):
        ax.text(x, y, txt, color=color, fontsize=fs, fontweight='bold',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=VAL_BG,
                          edgecolor=color, lw=1.4))

    def draw_cyl(cx, cy, w, h, title, tag):
        from matplotlib.patches import Ellipse
        n = 40
        for i in range(n):
            frac = i / n
            gray = 0.35 + 0.4 * np.sin(np.pi * frac)
            ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2 + h*i/n), w, h/n,
                                       color=(gray, gray, gray), zorder=2))
        for xc in [cx-w/2, cx+w/2]:
            ax.add_patch(Ellipse((xc, cy), width=h*0.4, height=h,
                                 facecolor='#7a7a7a', edgecolor='#333', lw=1.5, zorder=4))
        ax.add_patch(plt.Rectangle((cx-w/2, cy-h/2), w, h,
                                   fill=False, edgecolor='#444', lw=1.5, zorder=5))
        ax.text(cx, cy+0.15, title, color='white', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])
        ax.text(cx, cy-0.18, tag, color='#cccccc', fontsize=7,
                ha='center', va='center', zorder=6)

    ax.text(11, 8.6, 'SULFUR BURNER & WASTE HEAT BOILER — Detailed Streams',
            color='white', fontsize=13, fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='#2a5a7a')])

    draw_cyl(6.0, 4.5, 3.5, 1.5, 'SULFUR BURNER', '401-AF-01')
    arrow(2.5, 3.8, 4.2, 4.2, C_SOUFRE, lw=3.0)
    flow_box(2.0, 3.8,
             f"SOUFRE LIQUIDE\nT = {T_s:.0f}°C\nQ = {S_kgm:.0f} kg/min\nPureté ≈ 99.9 %",
             C_SOUFRE)
    arrow(6.0, 6.5, 6.0, 5.25, C_AIR, lw=2.5)
    flow_box(6.2, 6.8,
             f"AIR COMPRIMÉ SEC\nT = {turbo_res.get('T_out_C',0):.0f}°C\n"
             f"Q = {Air_nm3h:.0f} Nm³/h\nΔP = {turbo_res.get('delta_P_mmCE',0):.0f} mmCE",
             C_AIR)
    arrow(7.4, 4.5, 10.2, 4.5, C_GAZ, lw=4.0)
    flow_box(10.3, 4.5,
             f"GAZ CHAUD\nT = {results['four']['T_out']:.0f}°C\n"
             f"SO₂ = {results['four']['SO2_pct']:.2f} %\n"
             f"O₂ = {results['four']['O2_pct']:.2f} %",
             VAL_FG2)
    draw_cyl(14.5, 4.5, 4.0, 1.8, 'WASTE HEAT BOILER', '401-AV-01')
    arrow(10.4, 4.5, 12.5, 4.5, C_GAZ, lw=4.0)
    arrow(16.5, 4.5, 19.0, 4.5, C_GAZ, lw=3.0)
    flow_box(19.5, 4.5,
             f"GAZ REFROIDI\nT = {results['chaudiere']['T_out']:.0f}°C\navant bypass",
             VAL_FG2)
    arrow(14.5, 2.8, 14.5, 3.6, C_EAU_ALIM, lw=2.0)
    flow_box(14.5, 2.7, "EAU ALIM. (BFW)\nT ≈ 105°C P ≈ 45 bar", C_EAU_ALIM)
    arrow(14.5, 5.4, 14.5, 7.2, C_VAP_HP, lw=2.5)
    flow_box(14.7, 7.5,
             f"VAPEUR HP\nT ≈ 420°C P ≈ 40 bar\nQ = {results['chaudiere']['steam_flow']:.1f} t/h\n"
             f"Puissance = {results['chaudiere']['power_mw']:.1f} MW",
             C_VAP_HP)
    ax.text(14.5, 4.9,
            f"BYPASS chaudière : {results['chaudiere']['bypass_pct']:.1f} %",
            color='#ffaa44', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a2a2a', edgecolor='#ffaa44'))
    return fig


# =====================================================================
# PAGE 4 — CONVERTISSEUR + TOURS D'ABSORPTION
# =====================================================================

def render_page_conv_absorption(results, T_in_lits_user):
    BG = '#5a5a5a'
    fig, ax = plt.subplots(figsize=(26, 13))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 24)
    ax.set_ylim(-0.5, 12.0)
    ax.axis('off')
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03)

    conv = results['convertisseur']
    jd02 = results.get('jd02', {})
    jd03 = results.get('jd03', {})

    CX_LITS = 13.0; LIT_W = 2.2; LIT_H = 0.85; GAP = 1.6
    Y_LIT4 = 10.5; Y_LIT3 = Y_LIT4 - GAP; Y_LIT2 = Y_LIT3 - GAP; Y_LIT1 = Y_LIT2 - GAP
    CX_JD02 = 18.5; CX_JD03 = 7.5; TOUR_W = 0.90; TOUR_H = 1.30

    def draw_path(x1, y1, x2, y2, color='#cc3333', lw=2.5, connectionstyle="arc3,rad=0"):
        arrow_props = dict(
            arrowstyle="->,head_width=0.35,head_length=0.40",
            color=color, linewidth=lw, mutation_scale=15, capstyle='round',
            connectionstyle=connectionstyle,
            path_effects=[pe.withStroke(linewidth=lw+1.5, foreground='#222222')])
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **arrow_props))

    def flow_box(x, y, txt, color=VAL_FG, fs=7.2, align='center'):
        ax.text(x, y, txt, color=color, fontsize=fs, fontweight='bold',
                ha=align, va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.35', facecolor=VAL_BG,
                          edgecolor=color, lw=1.4))

    def draw_bed(cx, cy, w, h, bed_num, t_in, t_out, conv_pct):
        from matplotlib.patches import Ellipse
        n = 25
        for i in range(n):
            frac = i / n
            gray = 0.45 + 0.30 * np.sin(np.pi * frac)
            ax.add_patch(plt.Rectangle(
                (cx-w/2, cy-h/2+h*i/n), w, h/n, color=(gray, gray, gray), zorder=2))
        for xc in [cx-w/2, cx+w/2]:
            ax.add_patch(Ellipse((xc, cy), width=h*0.35, height=h,
                                 facecolor='#6a6a6a', edgecolor='#333', lw=1, zorder=4))
        ax.add_patch(plt.Rectangle((cx-w/2, cy-h/2), w, h,
                                   fill=False, edgecolor='#444', lw=1.2, zorder=5))
        ax.text(cx, cy+0.13, f"BED {bed_num}", color='white', fontsize=9,
                fontweight='bold', ha='center', va='center', zorder=6,
                path_effects=[pe.withStroke(linewidth=1, foreground='black')])
        ax.text(cx-w/3, cy-0.10, f"{t_in:.0f}°C", color=VAL_FG3, fontsize=8.5,
                fontweight='bold', ha='center', va='center', zorder=6)
        ax.text(cx+w/3, cy-0.10, f"{t_out:.0f}°C",
                color=VAL_FG2 if t_out > 550 else VAL_FG, fontsize=8.5,
                fontweight='bold', ha='center', va='center', zorder=6)
        ax.text(cx, cy-0.32, f"τ = {conv_pct:.1f}%", color=VAL_FG, fontsize=7.5,
                fontweight='bold', ha='center', va='center', zorder=6)

    ax.text(12, 11.65, 'CONVERTER & ABSORPTION TOWERS — Detailed Streams',
            color='white', fontsize=14, fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=2, foreground='#2a5a7a')])

    draw_path(9.8, Y_LIT1, CX_LITS - LIT_W/2, Y_LIT1, color=C_GAZ, lw=3)
    flow_box(9.4, Y_LIT1,
             f"GAZ ENTRANT\n(depuis chaudière)\n"
             f"T = {T_in_lits_user[0]:.0f}°C\nSO₂ + O₂ + N₂", VAL_FG2)
    ax.text(10.5, Y_LIT1 + 0.30, f"T = {T_in_lits_user[0]:.0f}°C",
            color=VAL_FG3, fontsize=8, fontweight='bold', ha='center')
    ax.text(10.5, Y_LIT1 + 0.08, 'GAS IN', color='#ffaa66', fontsize=7, ha='center')

    draw_bed(CX_LITS, Y_LIT1, LIT_W, LIT_H, 1,
             conv['T_in_lits'][0], conv['T_out_lits'][0], conv['tau_lits'][0])
    draw_path(CX_LITS, Y_LIT1 + LIT_H/2 + 0.05,
              CX_LITS, Y_LIT2 - LIT_H/2 - 0.05, color=C_GAZ, lw=2.5)
    flow_box(CX_LITS + LIT_W/2 - 0.5, (Y_LIT1 + Y_LIT2) / 2,
             f"Sortie BED1\nT = {conv['T_out_lits'][0]:.0f}°C\n"
             f"τ acc. = {conv['tau_lits'][0]:.1f}%", VAL_FG2, fs=7)

    draw_bed(CX_LITS, Y_LIT2, LIT_W, LIT_H, 2,
             conv['T_in_lits'][1], conv['T_out_lits'][1], conv['tau_lits'][1])
    draw_path(CX_LITS, Y_LIT2 + LIT_H/2 + 0.05,
              CX_LITS, Y_LIT3 - LIT_H/2 - 0.05, color=C_GAZ, lw=2.5)
    flow_box(CX_LITS + LIT_W/2 - 0.5, (Y_LIT2 + Y_LIT3) / 2,
             f"Sortie BED2\nT = {conv['T_out_lits'][1]:.0f}°C\n"
             f"τ acc. = {conv['tau_lits'][1]:.1f}%", VAL_FG2, fs=7)

    draw_bed(CX_LITS, Y_LIT3, LIT_W, LIT_H, 3,
             conv['T_in_lits'][2], conv['T_out_lits'][2], conv['tau_lits'][2])
    draw_path(CX_LITS + LIT_W/2, Y_LIT3,
              CX_JD02 - TOUR_W/2, Y_LIT3, color=C_GAZ, lw=2.8)
    ax.text((CX_LITS + LIT_W/2 + CX_JD02 - TOUR_W/2) / 2, Y_LIT3 + 0.22,
            'BED3 → JD02', color='#ffaa44', fontsize=7.5, ha='center', fontweight='bold')
    flow_box(CX_JD02 - 1.8, Y_LIT3,
             f"T = {conv['T_out_lits'][2]:.0f}°C τ acc. = {conv['tau_lits'][2]:.1f}%",
             C_GAZ, fs=7)

    draw_absorption_tower(ax, cx=CX_JD02, cy=Y_LIT3, w=TOUR_W, h=TOUR_H,
                          title='-', tag='Tour Intermédiaire')
    ax.text(CX_JD02, Y_LIT3 + TOUR_H/2 + 1.6,
            'JD02 — INTER ABSORPTION TOWER', color='white', fontsize=8.5,
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#2a5a7a')])
    ax.annotate("", xy=(CX_JD02, Y_LIT3 + TOUR_H/2),
                xytext=(CX_JD02, Y_LIT3 + TOUR_H/2 + 0.35),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_ACIDE_IN, lw=1.8), zorder=9)
    flow_box(CX_JD02, Y_LIT3 + TOUR_H/2 + 0.55,
             f"H₂SO₄ IN\nT = {jd02.get('T_acid_in', 0):.0f}°C\nw ≈ 98.5%", C_ACIDE_IN)
    ax.annotate("", xy=(CX_JD02, Y_LIT3 - TOUR_H/2 - 0.30),
                xytext=(CX_JD02, Y_LIT3 - TOUR_H/2),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_ACIDE_OUT, lw=1.8), zorder=9)
    flow_box(CX_JD02, Y_LIT3 - TOUR_H/2 - 0.55,
             f"H₂SO₄ OUT\nT = {jd02.get('T_acid_out', 0):.0f}°C\n"
             f"w = {jd02.get('w_H2SO4_out', 0):.2f}%", C_ACIDE_OUT)
    flow_box(CX_JD02 - 0.8, Y_LIT4,
             f"GAZ OUT JD02\nT = {jd02.get('T_gas_out', 0):.0f}°C\n"
             f"η = {jd02.get('eff_abs', 0):.1f}%\n"
             f"{jd02.get('ppm_SO3_out', 0):.0f} ppm SO₃", C_AIR, fs=7)

    draw_path(CX_JD02, Y_LIT3 + TOUR_H/2 + 0.05, CX_JD02, Y_LIT4, color=C_AIR, lw=2.8)
    draw_path(CX_JD02, Y_LIT4, CX_LITS + LIT_W/2, Y_LIT4, color=C_AIR, lw=2.8)
    ax.text((CX_JD02 + CX_LITS + LIT_W/2) / 2, Y_LIT4 + 0.22,
            'JD02 → BED4', color='#88aaff', fontsize=7.5, ha='center', fontweight='bold')
    flow_box((CX_JD02 + CX_LITS + LIT_W/2) / 2, Y_LIT4,
             f"Gaz désulfuré\nT = {jd02.get('T_gas_out', 0):.0f}°C", C_AIR, fs=7)

    draw_bed(CX_LITS, Y_LIT4, LIT_W, LIT_H, 4,
             conv['T_in_lits'][3], conv['T_out_lits'][3], conv['tau_lits'][3])
    draw_path(CX_LITS - LIT_W/2, Y_LIT4,
              CX_JD03 + TOUR_W/2, Y_LIT4, color=C_GAZ, lw=2.8)
    ax.text((CX_LITS - LIT_W/2 + CX_JD03 + TOUR_W/2) / 2, Y_LIT4 + 0.22,
            'BED4 → JD03', color='#ffaa44', fontsize=7.5, ha='center', fontweight='bold')
    flow_box((CX_LITS - LIT_W/2 + CX_JD03 + TOUR_W/2) / 2 + 0.5, Y_LIT4,
             f"T = {conv['T_out_lits'][3]:.0f}°C τ acc. = {conv['tau_lits'][3]:.1f}%",
             C_GAZ, fs=7)

    draw_absorption_tower(ax, cx=CX_JD03, cy=Y_LIT4, w=TOUR_W, h=TOUR_H,
                          title='-', tag='Tour Finale')
    ax.text(CX_JD03, Y_LIT4 + 0.2,
            'JD03 — FINAL ABSORPTION TOWER', color='white', fontsize=8.5,
            fontweight='bold', ha='center',
            path_effects=[pe.withStroke(linewidth=1.5, foreground='#2a5a7a')])
    ax.annotate("", xy=(CX_JD03, Y_LIT4 + TOUR_H/2),
                xytext=(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.35),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_ACIDE_IN, lw=1.8), zorder=9)
    flow_box(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.45,
             f"H₂SO₄ IN\nT = {jd03.get('T_acid_in', 0):.0f}°C\nw ≈ 98.5%", C_ACIDE_IN)
    ax.annotate("", xy=(CX_JD03, Y_LIT4 - TOUR_H/2 - 0.30),
                xytext=(CX_JD03, Y_LIT4 - TOUR_H/2),
                arrowprops=dict(arrowstyle="->,head_width=0.12,head_length=0.10",
                                color=C_ACIDE_OUT, lw=1.8), zorder=9)
    flow_box(CX_JD03, Y_LIT4 - TOUR_H/2 - 0.5,
             f"H₂SO₄ OUT\nT = {jd03.get('T_acid_out', 0):.0f}°C\n"
             f"w = {jd03.get('w_H2SO4_out', 0):.2f}%", C_ACIDE_OUT)
    flow_box(CX_JD03 + 1.8, Y_LIT4,
             f"GAZ IN JD03\nT = {conv['T_out_lits'][3]:.0f}°C\nSO₃ résiduel", C_GAZ, fs=7)

    draw_path(CX_JD03, Y_LIT4 + TOUR_H/2 + 0.05,
              CX_JD03, Y_LIT4 + TOUR_H/2 + 1.0, color=C_AIR, lw=2.8)
    flow_box(CX_JD03, Y_LIT4 + TOUR_H/2 + 1.20,
             f"GAZ TRAITÉ\nT = {jd03.get('T_gas_out', 0):.0f}°C\n"
             f"η = {jd03.get('eff_abs', 0):.1f}%\n"
             f"{jd03.get('ppm_SO3_out', 0):.0f} ppm SO₃\n→ Cheminée",
             '#88aaff')
    return fig


# =====================================================================
# INTERFACE STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="Supervision - Sulfuric Acid Plant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background-color: #5a5a5a !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0.4rem 1rem 0 1rem !important; }
#MainMenu, footer, header { visibility: hidden; }
.title-bar {
    background: #3c3c3c; border-bottom: 2px solid #222;
    padding: 4px 12px; display: flex;
    justify-content: space-between; align-items: center;
    font-size: 11px; color: #bbb;
}
div[data-testid="stSlider"] label { color: #ffffff !important; font-size: 12px !important; }
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"],
div[data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {
    color: #ffcc44 !important;
}
.ctrl-panel {
    background: #3a3a3a;
    border-radius: 8px;
    padding: 10px 16px 6px 16px;
    margin-bottom: 8px;
    border: 1px solid #555;
}
.ctrl-title {
    color: #ffcc44;
    font-weight: bold;
    font-size: 13px;
    margin-bottom: 6px;
}
.bac-panel {
    background: #2a2a2a;
    border-radius: 8px;
    padding: 8px 14px 6px 14px;
    margin-bottom: 8px;
    border: 1px solid #666;
}
.bac-title {
    color: #FFD700;
    font-weight: bold;
    font-size: 12px;
    margin-bottom: 4px;
}
.acid-title {
    color: #66cc44;
    font-weight: bold;
    font-size: 12px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Barre de titre ────────────────────────────────────────────────────
st.markdown(f"""
<div class="title-bar">
  <span>Supervision — Sulfuric Acid Plant | Digital Twin</span>
  <span style="font-family:monospace">{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</span>
</div>
""", unsafe_allow_html=True)

# ── Initialisation TankSystem en session_state ────────────────────────
if 'tank_system' not in st.session_state:
    st.session_state.tank_system = TankSystem()

# ── TABS ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "GLOBAL PROCESS",
    "AIR TREATMENT",
    "BURNER & BOILER",
    "CONVERTER & ABSORPTION",
    "HEAT EXCHANGERS",
])
# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL PROCESS (avec panneaux repliables)
# ═══════════════════════════════════════════════════════════════════════

with tab1:

    # ================================================================
    # 1. PARAMÈTRES PRINCIPAUX DU PROCÉDÉ (toujours visibles)
    # ================================================================
    with st.expander("process settings", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            S_kgm = st.slider("Soufre (kg/min)", 500.0, 1500.0, DEFAULT_S, 10.0, key="g_s")
        with c2:
            T_s = st.slider("T soufre (°C)", 100.0, 160.0, DEFAULT_T_S, 1.0, key="g_ts")
        with c3:
            Air_nm3h = st.slider("Air (Nm³/h)", 250000.0, 500000.0, DEFAULT_AIR, 5000.0, key="g_air")
        with c4:
            T_air = st.slider("T air (°C)", 80.0, 180.0, DEFAULT_T_AIR, 1.0, key="g_tair")
        with c5:
            ratio_p = st.slider("Ratio air primaire", 0.3, 0.8, DEFAULT_RATIO, 0.01, key="g_ratio")

    # ================================================================
    # 2. PARAMÈTRES BACS SOUFRE (repliables)
    # ================================================================
    with st.expander("sulfur tank", expanded=False):
        bs1, bs2, bs3 = st.columns(3)
        with bs1:
            sulfur_in_1 = st.slider("Livraison bac S1 (m³/s)", 0.0, 0.05, 0.0, 0.001,
                                    format="%.3f", key="bs_in1")
        with bs2:
            sulfur_in_2 = st.slider("Livraison bac S2 (m³/s)", 0.0, 0.05, 0.0, 0.001,
                                    format="%.3f", key="bs_in2")
        with bs3:
            split_ratio = st.slider("Split bac1/bac2 → four", 0.1, 0.9, 0.5, 0.05, key="bs_split")

        bs4, bs5, bs6 = st.columns(3)
        with bs4:
            sulfur_T_in = st.slider("T soufre livré (°C)", 120.0, 150.0, 140.0, 1.0, key="bs_tin")
        with bs5:
            h_s1_init = st.slider("Niveau init. bac S1 (%)", 10.0, 100.0,
                                  st.session_state.tank_system.sulfur_tank_1.level_pct,
                                  1.0, key="bs_h1")
        with bs6:
            h_s2_init = st.slider("Niveau init. bac S2 (%)", 10.0, 100.0,
                                  st.session_state.tank_system.sulfur_tank_2.level_pct,
                                  1.0, key="bs_h2")

    # ================================================================
    # 3. PARAMÈTRES BAC ACIDE (repliables)
    # ================================================================
    with st.expander("acid tank", expanded=False):
        ba1, ba2, ba3, ba4 = st.columns(4)
        with ba1:
            acid_prod_m3s = st.slider("Production acide tours (m³/s)", 0.0, 0.10, 0.03, 0.001,
                                      format="%.3f", key="ba_prod")
        with ba2:
            acid_out_m3s = st.slider("Expédition acide (m³/s)", 0.0, 0.10, 0.025, 0.001,
                                     format="%.3f", key="ba_out")
        with ba3:
            acid_T_in = st.slider("T acide entrant tours (°C)", 30.0, 80.0, 55.0, 0.5, key="ba_tin")
        with ba4:
            dt_step = st.slider("Pas de temps simulation (s)", 10.0, 600.0, 60.0, 10.0, key="ba_dt")

    # ================================================================
    # 4. MISE À JOUR DU TANKSYSTEM ET SIMULATION
    # ================================================================
    ts = st.session_state.tank_system

    # Recalibrage des niveaux initiaux (si demandé)
    if abs(ts.sulfur_tank_1.level_pct - h_s1_init) > 2.0:
        ts.sulfur_tank_1.h_B = h_s1_init / 100.0 * ts.sulfur_tank_1.h_max
    if abs(ts.sulfur_tank_2.level_pct - h_s2_init) > 2.0:
        ts.sulfur_tank_2.h_B = h_s2_init / 100.0 * ts.sulfur_tank_2.h_max

    # Mise à jour des flux et avancement d'un pas
    ts.update_flows(
        S_kgmin=S_kgm,
        acid_prod_m3s=acid_prod_m3s,
        acid_out_m3s=acid_out_m3s,
        sulfur_in_1_m3s=sulfur_in_1,
        sulfur_in_2_m3s=sulfur_in_2,
        split_ratio=split_ratio,
        acid_T_in=acid_T_in,
        sulfur_T_in=sulfur_T_in,
    )
    ts.step(dt=float(dt_step))

    # ================================================================
    # 5. SIMULATION DU PROCÉDÉ (flowsheet)
    # ================================================================
    with st.spinner("Calcul en cours..."):
        results1 = simuler_complet(S_kgm, Air_nm3h, ratio_p, T_air, T_s)

        P_ATM = 101325.0
        af1 = AirFilter()
        af1.Q_gas_Nm3h = Air_nm3h
        af1.T_in = 30.0
        af1.P_in = P_ATM
        try:
            fr1 = af1.compute(t_hours=0.0)
            if 'P_in_Pa' not in fr1:
                fr1['P_in_Pa'] = P_ATM
            Taf1 = fr1['T_out']
        except:
            fr1 = {'delta_P_mmWC': 0.0, 'T_out': 30.0, 'P_in_Pa': P_ATM, 'P_out_Pa': P_ATM}
            Taf1 = 30.0

        dt1 = DryingTower()
        dt1.Q_gas_Nm3h = Air_nm3h
        dt1.Q_acid_m3h = 1245.0
        dt1.T_acid_in = 50.0
        dt1.T_gas_in = Taf1
        try:
            dr1 = dt1.compute()
            dr1['w_H2SO4_in'] = dt1.w_H2SO4_in * 100.0
        except:
            dr1 = {'TG_out': Taf1, 'TL_out': 50.0, 'eff': 0.0,
                   'w_gNm3': 0.0, 'w_H2SO4_in': 98.6}

        tr1 = TurboBlowerTrain()
        tr1.Q_gas_Nm3h = Air_nm3h
        tr1.T_in = dr1.get('TG_out', Taf1)
        turbo1 = tr1.compute(load=1.0)

    # ================================================================
    # 6. AFFICHAGE DU FLOWSHEET
    # ================================================================
    if results1:
        fig1 = draw_flowsheet(results1, S_kgm, Air_nm3h, T_air, T_s, ratio_p,
                              turbo1, dr1, fr1, ts)
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", dpi=150, bbox_inches='tight',
                     pad_inches=0.05, facecolor=fig1.get_facecolor())
        buf1.seek(0)
        st.markdown(
            f'<img src="data:image/png;base64,{base64.b64encode(buf1.read()).decode()}" '
            f'style="width:100%;min-height:78vh;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
        plt.close(fig1)
    else:
        st.error("Erreur de simulation.")

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — AIR TREATMENT
# ══════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ Control Panel — Air Treatment</div>',
                unsafe_allow_html=True)
    ca1, ca2, ca3, ca4, ca5 = st.columns(5)
    with ca1: Air_t2    = st.slider("Débit air (Nm³/h)",   250000.0, 500000.0, DEFAULT_AIR, 5000.0, key="at_air")
    with ca2: T_air_t2  = st.slider("T air ambiant (°C)",      10.0,     50.0,       30.0,    1.0, key="at_tair")
    with ca3: P_air_kPa = st.slider("Pression air (kPa)",       95.0,    110.0,      101.3,    0.1, key="at_p")
    with ca4: Q_acid_t2 = st.slider("Débit acide (m³/h)",      800.0,  1800.0,     1245.0,   10.0, key="at_qa")
    with ca5: T_acid_t2 = st.slider("T acide entrée (°C)",      40.0,     75.0,       50.0,    0.5, key="at_ta")
    st.markdown('</div>', unsafe_allow_html=True)

    P_ATM2 = P_air_kPa * 1000.0
    af2 = AirFilter(); af2.Q_gas_Nm3h = Air_t2; af2.T_in = T_air_t2; af2.P_in = P_ATM2
    try:
        fr2 = af2.compute(t_hours=0.0)
        if 'P_in_Pa' not in fr2: fr2['P_in_Pa'] = P_ATM2
        Taf2 = fr2['T_out']
    except:
        fr2 = {'delta_P_mmWC': 0.0, 'T_out': T_air_t2, 'P_in_Pa': P_ATM2, 'P_out_Pa': P_ATM2}
        Taf2 = T_air_t2

    dt2 = DryingTower(); dt2.Q_gas_Nm3h = Air_t2; dt2.Q_acid_m3h = Q_acid_t2
    dt2.T_acid_in = T_acid_t2; dt2.T_gas_in = Taf2
    try:
        dr2 = dt2.compute(); dr2['w_H2SO4_in'] = dt2.w_H2SO4_in * 100.0
    except:
        dr2 = {'TG_out': Taf2, 'TL_out': T_acid_t2, 'eff': 0.0,
               'w_gNm3': 0.0, 'w_H2SO4_in': 98.6}

    tr2 = TurboBlowerTrain(); tr2.Q_gas_Nm3h = Air_t2; tr2.T_in = dr2.get('TG_out', Taf2)
    turbo2 = tr2.compute(load=1.0)

    fig2 = render_page_equipements(fr2, dr2, turbo2, Air_t2, 0.0, Q_acid_t2, T_acid_t2, P_air_kPa)
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=140, bbox_inches='tight',
                 pad_inches=0.05, facecolor=fig2.get_facecolor())
    buf2.seek(0)
    st.markdown(
        f'<img src="data:image/png;base64,{base64.b64encode(buf2.read()).decode()}" '
        f'style="width:100%;object-fit:contain;display:block;" />',
        unsafe_allow_html=True)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — BURNER & BOILER
# ══════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ Control Panel — Sulfur Burner & Boiler</div>',
                unsafe_allow_html=True)
    cb1, cb2, cb3, cb4, cb5, cb6 = st.columns(6)
    with cb1: S_t3     = st.slider("Soufre (kg/min)",    500.0, 1500.0, DEFAULT_S,      10.0, key="fb_s")
    with cb2: Ts_t3    = st.slider("T soufre (°C)",      100.0,  160.0, DEFAULT_T_S,     1.0, key="fb_ts")
    with cb3: Air_t3   = st.slider("Débit air (Nm³/h)", 250000.0, 500000.0, DEFAULT_AIR, 5000.0, key="fb_air")
    with cb4: Tair_t3  = st.slider("T air (°C)",          80.0,  180.0, DEFAULT_T_AIR,   1.0, key="fb_tair")
    with cb5: ratio_t3 = st.slider("Ratio air 2nd",        0.2,    0.7, 1.0-DEFAULT_RATIO, 0.01, key="fb_ratio")
    with cb6: T_cible  = st.slider("T cible conv. (°C)",  380.0,  450.0, float(T_TARGET_CONV), 1.0, key="fb_tcible")
    st.markdown('</div>', unsafe_allow_html=True)

    ratio_p_t3 = 1.0 - ratio_t3
    with st.spinner("Calcul four + chaudière..."):
        res3 = simuler_complet(S_t3, Air_t3, ratio_p_t3, Tair_t3, Ts_t3)

    tr3 = TurboBlowerTrain(); tr3.Q_gas_Nm3h = Air_t3; tr3.T_in = 30.0
    turbo3 = tr3.compute(load=1.0)

    if res3:
        fig3 = render_page_four_chaudiere(res3, S_t3, Ts_t3, Air_t3, Tair_t3, turbo3)
        buf3 = io.BytesIO()
        fig3.savefig(buf3, format="png", dpi=140, bbox_inches='tight',
                     pad_inches=0.05, facecolor=fig3.get_facecolor())
        buf3.seek(0)
        st.markdown(
            f'<img src="data:image/png;base64,{base64.b64encode(buf3.read()).decode()}" '
            f'style="width:100%;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
        plt.close(fig3)
    else:
        st.error("Erreur de simulation.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — CONVERTER & ABSORPTION
# ══════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="ctrl-panel"><div class="ctrl-title">⚙ Control Panel — Converter & Absorption</div>',
                unsafe_allow_html=True)
    cc1, cc2, cc3, cc4, cc5 = st.columns(5)
    with cc1: T_in_lit1 = st.slider("T entrée BED 1 (°C)", 380.0, 450.0, float(T_TARGET_CONV), 1.0, key="cv_t1")
    with cc2: T_in_lit2 = st.slider("T entrée BED 2 (°C)", 400.0, 480.0, 454.0, 1.0, key="cv_t2")
    with cc3: T_in_lit3 = st.slider("T entrée BED 3 (°C)", 400.0, 470.0, 449.0, 1.0, key="cv_t3")
    with cc4: T_in_lit4 = st.slider("T entrée BED 4 (°C)", 390.0, 450.0, 425.0, 1.0, key="cv_t4")
    with cc5: S_t4      = st.slider("Soufre (kg/min)",      500.0, 1500.0, DEFAULT_S, 10.0, key="cv_s")
    st.markdown('</div>', unsafe_allow_html=True)

    T_in_lits_user = [T_in_lit1, T_in_lit2, T_in_lit3, T_in_lit4]
    with st.spinner("Calcul convertisseur + absorption..."):
        res4 = simuler_complet(S_t4, DEFAULT_AIR, DEFAULT_RATIO, DEFAULT_T_AIR, DEFAULT_T_S)

    if res4:
        res4['convertisseur']['T_in_lits'] = T_in_lits_user
        fig4 = render_page_conv_absorption(res4, T_in_lits_user)
        buf4 = io.BytesIO()
        fig4.savefig(buf4, format="png", dpi=140, bbox_inches='tight',
                     pad_inches=0.05, facecolor=fig4.get_facecolor())
        buf4.seek(0)
        st.markdown(
            f'<img src="data:image/png;base64,{base64.b64encode(buf4.read()).decode()}" '
            f'style="width:100%;object-fit:contain;display:block;" />',
            unsafe_allow_html=True)
        plt.close(fig4)
    else:
        st.error("Erreur de simulation.")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — HEAT EXCHANGERS
# ══════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown(
        '<div class="ctrl-panel"><div class="ctrl-title">'
        '⚙ Control Panel — Exchangers'
        '</div>',
        unsafe_allow_html=True,
    )
    hx_c1, hx_c2, hx_c3, hx_c4, hx_c5, hx_c6 = st.columns(6)
    with hx_c1:
        hx_S = st.slider("Soufre (kg/min)", 500.0, 1500.0, DEFAULT_S, 10.0, key="hx_s")
    with hx_c2:
        hx_air = st.slider("Air (Nm³/h)", 250_000.0, 500_000.0, DEFAULT_AIR, 5_000.0, key="hx_air")
    with hx_c3:
        hx_T_sh_gas_in = st.slider("T gaz → SH1B (°C)", 580.0, 680.0, 627.0, 1.0, key="hx_sh_tin")
    with hx_c4:
        hx_T_hi_gas_in = st.slider("T gaz → Hot IHX (°C)", 480.0, 560.0, 516.0, 1.0, key="hx_hi_tin")
    with hx_c5:
        hx_T_ci_gas_in = st.slider("T gaz → Cold IHX (°C)", 420.0, 500.0, 460.0, 1.0, key="hx_ci_tin")
    with hx_c6:
        hx_T_bed4_out = st.slider("T gaz → HP4A (°C)", 360.0, 450.0, 406.0, 1.0, key="hx_bed4_out")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("Calcul des échangeurs…"):
        _res_hx = simuler_complet(hx_S, hx_air, DEFAULT_RATIO, DEFAULT_T_AIR, DEFAULT_T_S)

        _comp_interpass = {'SO2': 0.05, 'SO3': 0.02, 'O2': 0.09, 'N2': 0.84}
        _comp_final     = {'SO2': 0.003, 'SO3': 0.05, 'O2': 0.10, 'N2': 0.847}
        _mdot_gas = hx_S / 60.0 * 2.0 * 5.5

        def _run_hx(cls, T_gas_in_C, P_gas=150_000.0, comp=None):
            hx = cls()
            hx.set_gas_inlet(
                T_C=T_gas_in_C,
                P_Pa=P_gas,
                mdot_kg_s=_mdot_gas,
                composition=(comp or _comp_interpass),
            )
            return hx.compute()

        r_sh  = _run_hx(HPSuperheater1B,  hx_T_sh_gas_in)
        r_hi  = _run_hx(HotInterpassHX,   hx_T_hi_gas_in)
        r_ci  = _run_hx(ColdInterpassHX,  hx_T_ci_gas_in)
        r_ec  = _run_hx(Economizer3B,     r_ci.get('T_gas_out_C', 306.0))
        r_hp4a = _run_hx(HP4AExchanger,   hx_T_bed4_out,              comp=_comp_final)
        r_lp4a = _run_hx(LP4AExchanger,   r_hp4a.get('T_gas_out_C', 340.0), comp=_comp_final)
        r_e4c  = _run_hx(E4CExchanger,    r_lp4a.get('T_gas_out_C', 270.0), comp=_comp_final)
        r_e4a  = _run_hx(E4AExchanger,    r_e4c.get('T_gas_out_C', 200.0),  comp=_comp_final)

        hx_results = {
            'hp_superheater_1b': r_sh,
            'hot_interpass':     r_hi,
            'cold_interpass':    r_ci,
            'economizer_3b':     r_ec,
            'hp4a':  r_hp4a,
            'lp4a':  r_lp4a,
            'e4c':   r_e4c,
            'e4a':   r_e4a,
        }

        _conv_res_hx = _res_hx.get('convertisseur', {}) if _res_hx else {}
        if 'dp_lits' not in _conv_res_hx:
            _conv_res_hx['dp_lits'] = [1.69, 1.75, 2.14, 1.90]

        fig5 = render_page_exchangers(hx_results, conv_results=_conv_res_hx)

    buf5 = io.BytesIO()
    fig5.savefig(buf5, format="png", dpi=150, bbox_inches="tight",
                 pad_inches=0.05, facecolor=fig5.get_facecolor())
    buf5.seek(0)
    st.markdown(
        f'<img src="data:image/png;base64,{base64.b64encode(buf5.read()).decode()}" '
        f'style="width:100%;min-height:75vh;object-fit:contain;display:block;" />',
        unsafe_allow_html=True,
    )
    plt.close(fig5)