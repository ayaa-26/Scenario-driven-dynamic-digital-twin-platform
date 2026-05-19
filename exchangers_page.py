"""
exchangers_page.py — Page Échangeurs style UniSim-Design
Refonte complète : convertisseur tour grise, échangeurs cylindres métalliques,
palette UniSim, blocs de données noirs/verts, fond gris clair #B0B0B0.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════
# PALETTE STYLE UNISIM (cohérente avec app.py)
# ══════════════════════════════════════════════════════════════════════
BG_MAIN      = '#B0B0B0'
BG_DARK      = '#3C3C3C'
BG_DATA      = '#000000'
FG_DATA      = '#00FF44'
FG_DATA2     = '#FFAA00'
FG_DATA3     = '#FFEE00'
FG_UNIT      = '#88FFBB'
METAL_LIGHT  = '#E8E8E8'
METAL_MID    = '#B8B8B8'
METAL_DARK   = '#888888'
METAL_BORDER = '#606060'

C_GAZ        = '#DD8500'    # Gaz process / SO2 (jaune-orange)
C_VAP_HP     = '#DD2222'    # Vapeur HP (rouge)
C_VAP_BP     = '#FF9999'    # Vapeur BP (rouge clair)
C_EAU_ALIM   = '#3366CC'    # Eau alimentaire (bleu)
C_RECYCLE    = '#CCEE00'    # Recycle SO2 (vert-jaune)
C_ACIDE_IN   = '#22AA22'    # Acide entrant (vert)
C_ACIDE_OUT  = '#66DD44'    # Acide sortant (vert clair)
C_ACID_VIOLET= '#9933CC'    # Acide H2SO4 (violet)


# ══════════════════════════════════════════════════════════════════════
# HELPERS GRAPHIQUES STYLE UNISIM
# ══════════════════════════════════════════════════════════════════════

def _unisim_bg(fig, ax, xlim=(0, 52), ylim=(-4.5, 22.0)):
    fig.patch.set_facecolor(BG_MAIN)
    ax.set_facecolor(BG_MAIN)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')
    fig.subplots_adjust(left=0.006, right=0.994, top=0.965, bottom=0.015)


def _title_bar(ax, title, xlim=52, ylim=22.0):
    ax.add_patch(plt.Rectangle((0, ylim - 0.60), xlim, 0.60,
                               facecolor=BG_DARK, edgecolor='none', zorder=20))
    ax.text(xlim / 2, ylim - 0.30, title,
            color='white', fontsize=10, fontweight='bold',
            ha='center', va='center', zorder=21, fontfamily='monospace')
    for i, (lbl, col) in enumerate([('×', '#CC3333'), ('□', '#888'), ('─', '#888')]):
        bx = xlim - 0.60 * (i + 1)
        ax.add_patch(plt.Rectangle((bx - 0.24, ylim - 0.50), 0.48, 0.38,
                                   facecolor=col if i == 0 else '#555',
                                   edgecolor='#333', lw=0.5, zorder=22))
        ax.text(bx, ylim - 0.31, lbl, color='white', fontsize=7,
                ha='center', va='center', zorder=23)


def _data_box(ax, x, y, lines, w=1.80):
    """Boîte de données noire avec texte vert fluo style UniSim."""
    lh = 0.28
    h = len(lines) * lh + 0.10
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                               boxstyle="square,pad=0",
                               facecolor=BG_DATA, edgecolor='#444', lw=0.8, zorder=15))
    for i, line in enumerate(lines):
        yy = y + h - (i + 1) * lh + 0.05
        if isinstance(line, tuple) and len(line) == 2:
            val, unit = line
            ax.text(x + w - 0.22, yy, val, color=FG_DATA,
                    fontsize=7.5, fontweight='bold', ha='right', va='center',
                    fontfamily='monospace', zorder=16)
            ax.text(x + w - 0.06, yy, unit, color=FG_UNIT,
                    fontsize=6, ha='right', va='center',
                    fontfamily='monospace', zorder=16)
        elif isinstance(line, tuple) and len(line) == 3:
            val, unit, color = line
            ax.text(x + w - 0.22, yy, val, color=color,
                    fontsize=7.5, fontweight='bold', ha='right', va='center',
                    fontfamily='monospace', zorder=16)
            ax.text(x + w - 0.06, yy, unit, color=FG_UNIT,
                    fontsize=6, ha='right', va='center',
                    fontfamily='monospace', zorder=16)
        else:
            txt = line[0] if isinstance(line, tuple) else line
            ax.text(x + 0.08, yy, txt, color=FG_DATA,
                    fontsize=7, fontweight='bold', ha='left', va='center',
                    fontfamily='monospace', zorder=16)


def _stream_tag(ax, x, y, name):
    """Étiquette de flux style UniSim."""
    w = len(name) * 0.090 + 0.22
    h = 0.24
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="square,pad=0.02",
                               facecolor='#CCCCCC', edgecolor='#888', lw=0.7, zorder=18))
    ax.text(x, y, name, color='#111', fontsize=6.5, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=19)


def _pipe(ax, x1, y1, x2, y2, color=C_GAZ, lw=2.5, arrow=True, tag=None):
    """Tuyau à angles droits avec flèche directionnelle."""
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
                frac = 0.5
                ax.annotate("",
                    xy=(x1 + dx * (frac + 0.005), y1 + dy * (frac + 0.005)),
                    xytext=(x1 + dx * (frac - 0.005), y1 + dy * (frac - 0.005)),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.18",
                                   color=color, lw=lw), zorder=11)
    if tag:
        tx = (x1 + x2) / 2
        ty = (y1 + y2) / 2
        _stream_tag(ax, tx, ty + 0.16, tag)


def _arr(ax, x1, y1, x2, y2, color, lw=2.2, zo=10):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->,head_width=0.26,head_length=0.20",
                                color=color, lw=lw),
                zorder=zo)


def _line(ax, xs, ys, color, lw=2.2, zo=10):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='butt', zorder=zo)


# ══════════════════════════════════════════════════════════════════════
# ÉCHANGEUR TUBULAIRE HORIZONTAL — STYLE UNISIM MÉTALLIQUE
# ══════════════════════════════════════════════════════════════════════

def _hx_unisim(ax, cx, cy, w, h, label='', tag=''):
    """
    Cylindre horizontal gris métallisé avec embouts latéraux.
    Style P&ID industriel UniSim.
    """
    # Corps principal avec dégradé métallique
    n = 30
    for i in range(n):
        frac = i / n
        if frac < 0.5:
            t = frac * 2
            gray = tuple(
                np.array(mcolors.to_rgb(METAL_DARK)) * (1 - t) +
                np.array(mcolors.to_rgb(METAL_LIGHT)) * t)
        else:
            t = (frac - 0.5) * 2
            gray = tuple(
                np.array(mcolors.to_rgb(METAL_LIGHT)) * (1 - t) +
                np.array(mcolors.to_rgb(METAL_DARK)) * t)
        ax.add_patch(plt.Rectangle(
            (cx - w / 2, cy - h / 2 + h * i / n), w, h / n,
            color=gray, zorder=3))

    # Bordure principale
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                               fill=False, edgecolor=METAL_BORDER, lw=2.0, zorder=5))

    # Embouts latéraux (têtes de distribution)
    cap_w = h * 0.28
    cap_h = h * 1.10
    for side in [-1, 1]:
        cx_cap = cx + side * (w / 2 + cap_w / 2)
        ax.add_patch(plt.Rectangle(
            (cx_cap - cap_w / 2, cy - cap_h / 2), cap_w, cap_h,
            facecolor=METAL_MID, edgecolor=METAL_BORDER, lw=1.5, zorder=4))
        # Ellipse extrémité
        ax.add_patch(Ellipse(
            (cx_cap + side * cap_w / 2, cy),
            width=cap_w * 0.5, height=cap_h,
            facecolor=METAL_DARK, edgecolor=METAL_BORDER, lw=1.2, zorder=5))

    # Lignes internes symbolisant les tubes
    n_tubes = 4
    for i in range(1, n_tubes + 1):
        yy = cy - h / 2 + h * i / (n_tubes + 1)
        ax.plot([cx - w / 2 + 0.08, cx + w / 2 - 0.08], [yy, yy],
                color='#999999', lw=0.7, zorder=5, alpha=0.6)

    # Baffles internes
    for frac_b in [0.33, 0.67]:
        xx = cx - w / 2 + w * frac_b
        ax.plot([xx, xx], [cy - h / 2 + 0.05, cy + h / 2 - 0.05],
                color='#888', lw=1.0, zorder=5, alpha=0.5)

    # Labels
    ax.text(cx, cy + 0.06, label, color='#111111', fontsize=8, fontweight='bold',
            ha='center', va='center', fontfamily='monospace', zorder=7)
    if tag:
        ax.text(cx, cy - 0.16, tag, color='#333333', fontsize=6.5,
                ha='center', va='center', fontfamily='monospace', zorder=7)


# ══════════════════════════════════════════════════════════════════════
# SURCHAUFFEUR VERTICAL STYLE UNISIM
# ══════════════════════════════════════════════════════════════════════

def _superheater_unisim(ax, cx, cy_bot, w, h, label):
    """Surchauffeur vertical gris métallisé avec dôme bleu style UniSim."""
    # Corps principal avec dégradé
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
            (cx - w / 2, cy_bot + h * i / n), w, h / n,
            color=gray, zorder=3))

    # Bordure
    ax.add_patch(plt.Rectangle((cx - w / 2, cy_bot), w, h,
                               fill=False, edgecolor=METAL_BORDER, lw=2.0, zorder=5))

    # Embouts horizontaux (haut et bas)
    cap_h = h * 0.10
    cap_w = w * 1.20
    for side_y in [cy_bot - cap_h / 2, cy_bot + h - cap_h / 2]:
        ax.add_patch(plt.Rectangle((cx - cap_w / 2, side_y), cap_w, cap_h,
                                   facecolor=METAL_MID, edgecolor=METAL_BORDER,
                                   lw=1.2, zorder=4))

    # Ellipses extrémités verticales
    for yc in [cy_bot, cy_bot + h]:
        ax.add_patch(Ellipse((cx, yc), width=w, height=w * 0.28,
                             facecolor=METAL_MID, edgecolor=METAL_BORDER,
                             lw=1.2, zorder=5))

    # Dôme bleu en haut
    dome_ry = w * 0.35
    ax.add_patch(Ellipse((cx, cy_bot + h + dome_ry * 0.7),
                         w * 0.95, dome_ry * 2.0,
                         facecolor='#2244BB', edgecolor='#88AAFF', lw=1.8, zorder=5))

    # Lignes internes tubes
    for i in range(1, 5):
        tx = cx - w / 2 + w * i / 5
        ax.plot([tx, tx], [cy_bot + 0.08, cy_bot + h - 0.08],
                color='#999', lw=0.6, zorder=5, alpha=0.5)

    # Label
    ax.text(cx, cy_bot + h * 0.50, label, color='#111', fontsize=7.5,
            fontweight='bold', ha='center', va='center',
            fontfamily='monospace', zorder=7)


# ══════════════════════════════════════════════════════════════════════
# TOUR D'ABSORPTION — STYLE UNISIM
# ══════════════════════════════════════════════════════════════════════

def _tower_unisim(ax, cx, cy, w, h, label, tag=''):
    """Tour d'absorption style UniSim avec garnissage en X."""
    # Corps avec dégradé gris
    n = 20
    for i in range(n):
        frac = i / n
        g = 0.60 + 0.25 * np.sin(np.pi * frac)
        ax.add_patch(plt.Rectangle(
            (cx - w / 2, cy - h / 2 + h * i / n), w, h / n,
            color=(g, g, g * 0.9), zorder=3))
    ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                               fill=False, edgecolor=METAL_BORDER, lw=1.8, zorder=5))

    # Ellipses extrémités
    for yc in [cy - h / 2, cy + h / 2]:
        ax.add_patch(Ellipse((cx, yc), width=w, height=w * 0.25,
                             facecolor=METAL_MID, edgecolor=METAL_BORDER,
                             lw=1.0, zorder=4))

    # Garnissage X interne
    pack_y1 = cy - h / 2 + h * 0.12
    pack_y2 = cy + h / 2 - h * 0.18
    n_levels = 5
    for i in range(n_levels + 1):
        yy = pack_y1 + (pack_y2 - pack_y1) * i / n_levels
        ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [yy, yy],
                color='#888', lw=0.5, zorder=6, alpha=0.7)
    ax.plot([cx - w / 2 + 0.04, cx + w / 2 - 0.04], [pack_y1, pack_y2],
            color='#777', lw=0.7, zorder=6, alpha=0.6)
    ax.plot([cx + w / 2 - 0.04, cx - w / 2 + 0.04], [pack_y1, pack_y2],
            color='#777', lw=0.7, zorder=6, alpha=0.6)

    # Distributeur liquide
    dist_y = cy + h / 2 - h * 0.22
    ax.plot([cx - w / 2 + 0.05, cx + w / 2 - 0.05], [dist_y, dist_y],
            color=C_ACID_VIOLET, lw=1.5, zorder=7)
    for xi in np.linspace(cx - w / 2 + 0.07, cx + w / 2 - 0.07, 3):
        ax.plot([xi, xi], [dist_y, dist_y - 0.06], color=C_ACID_VIOLET, lw=1.0, zorder=7)

    # Labels
    ax.text(cx, cy + h / 2 + 0.24, label,
            color='#111', fontsize=8, fontweight='bold',
            ha='center', va='bottom', fontfamily='monospace', zorder=8)
    if tag:
        ax.text(cx, cy - h / 2 - 0.16, tag,
                color='#333', fontsize=6.5,
                ha='center', va='top', fontfamily='monospace', zorder=8)


# ══════════════════════════════════════════════════════════════════════
# CONVERTISSEUR — 4 LITS — TOUR GRISE AVEC RELIEF UNISIM
# ══════════════════════════════════════════════════════════════════════

def _converter_unisim(ax, cx, cy_bot, bed_w, bed_h, gap, conv):
    """
    Convertisseur style UniSim : tour grise avec 4 lits empilés,
    dégradés métalliques, données par lit sur fond vert foncé/noir.
    """
    T_in  = conv.get('T_in_lits',  [420., 527., 440., 393.])
    T_out = conv.get('T_out_lits', [627., 516., 460., 406.])
    tau   = conv.get('tau_lits',   [65.92, 73.25, 64.48, 98.99])
    dp    = conv.get('dp_lits',    [1.69, 1.75, 2.14, 1.90])
    n = 4
    total_h = n * bed_h + (n - 1) * gap
    pad = 0.30

    # Enveloppe extérieure — tour grise avec dégradé métallique
    nb_grad = 40
    for i in range(nb_grad):
        frac = i / nb_grad
        if frac < 0.12:
            gray = METAL_DARK
        elif frac < 0.45:
            t = (frac - 0.12) / 0.33
            gray = tuple(np.array(mcolors.to_rgb(METAL_DARK)) * (1 - t) +
                         np.array(mcolors.to_rgb(METAL_LIGHT)) * t)
        elif frac < 0.88:
            t = (frac - 0.45) / 0.43
            gray = tuple(np.array(mcolors.to_rgb(METAL_LIGHT)) * (1 - t) +
                         np.array(mcolors.to_rgb(METAL_MID)) * t)
        else:
            gray = METAL_DARK
        ax.add_patch(plt.Rectangle(
            (cx - bed_w / 2 - pad, cy_bot - pad + (total_h + 2 * pad) * i / nb_grad),
            bed_w + 2 * pad, (total_h + 2 * pad) / nb_grad,
            color=gray, zorder=2))

    # Bordure extérieure
    ax.add_patch(plt.Rectangle(
        (cx - bed_w / 2 - pad, cy_bot - pad),
        bed_w + 2 * pad, total_h + 2 * pad,
        fill=False, edgecolor=METAL_BORDER, lw=2.5, zorder=6))

    # Ellipses haut/bas style cylindre
    for yc, z in [(cy_bot - pad, 3), (cy_bot + total_h + pad, 7)]:
        ax.add_patch(Ellipse(
            (cx, yc), width=bed_w + 2 * pad, height=(bed_w + 2 * pad) * 0.15,
            facecolor=METAL_MID, edgecolor=METAL_BORDER, lw=1.5, zorder=z))

    # Titre convertisseur
    ax.text(cx, cy_bot + total_h + pad + 0.50,
            'CONVERTISSEUR', color='#111', fontsize=11, fontweight='bold',
            ha='center', va='bottom', fontfamily='monospace', zorder=13)
    ax.text(cx, cy_bot + total_h + pad + 0.22,
            '401AV02', color='#333', fontsize=8,
            ha='center', va='bottom', fontfamily='monospace', zorder=13)

    bed_cy = []
    for i in range(n):
        cy_bed = cy_bot + i * (bed_h + gap)
        mid = cy_bed + bed_h / 2
        bed_cy.append(mid)

        # Corps du lit avec dégradé bleuté
        nb = 20
        for b in range(nb):
            fr = b / nb
            gv = 0.55 + 0.25 * np.sin(np.pi * fr)
            ax.add_patch(plt.Rectangle(
                (cx - bed_w / 2, cy_bed + bed_h * b / nb),
                bed_w, bed_h / nb,
                color=(gv * 0.78, gv * 0.82, gv), zorder=3))

        # Bordure du lit
        ax.add_patch(plt.Rectangle(
            (cx - bed_w / 2, cy_bed), bed_w, bed_h,
            fill=False, edgecolor='#4466AA', lw=1.5, zorder=4))

        # Hachures X internes
        ax.plot([cx - bed_w / 2 + 0.08, cx + bed_w / 2 - 0.08],
                [cy_bed + 0.08, cy_bed + bed_h - 0.08],
                color='#8899BB', lw=0.7, zorder=5, alpha=0.5)
        ax.plot([cx + bed_w / 2 - 0.08, cx - bed_w / 2 + 0.08],
                [cy_bed + 0.08, cy_bed + bed_h - 0.08],
                color='#8899BB', lw=0.7, zorder=5, alpha=0.5)

        # Nom du lit
        ax.text(cx, cy_bed + bed_h * 0.78,
                f"COUCHE {i + 1}", color='#111', fontsize=9, fontweight='bold',
                ha='center', va='center', fontfamily='monospace', zorder=6)

        # Bloc données internes — X sur fond vert foncé
        bx_data = cx - bed_w / 2 + 0.08
        bw_data = bed_w * 0.46 - 0.08
        bh_data = bed_h * 0.30
        by_data = cy_bed + bed_h * 0.14

        ax.add_patch(FancyBboxPatch((bx_data, by_data), bw_data, bh_data,
                                   boxstyle="square,pad=0",
                                   facecolor='#003300', edgecolor='#005500', lw=0.8, zorder=6))
        ax.text(bx_data + bw_data / 2, by_data + bh_data * 0.65,
                f"X={tau[i]:.2f}%", color='#00FF44', fontsize=7.5, fontweight='bold',
                ha='center', va='center', fontfamily='monospace', zorder=7)
        ax.text(bx_data + bw_data / 2, by_data + bh_data * 0.22,
                f"ΔP={dp[i]:.2f}kPa", color=FG_DATA3, fontsize=7,
                ha='center', va='center', fontfamily='monospace', zorder=7)

        # Boîtes de données Tin/Tout — style UniSim noir/vert
        # À gauche : T entrée
        _data_box(ax, cx - bed_w / 2 - 2.05, mid - 0.20, [
            (f"{T_in[i]:.0f}", "°C in", FG_DATA3),
        ], w=1.80)
        # À droite : T sortie
        _data_box(ax, cx + bed_w / 2 + 0.20, mid - 0.20, [
            (f"{T_out[i]:.0f}", "°C out", FG_DATA2),
        ], w=1.80)

        # Séparateur entre lits
        if i < n - 1:
            sep_y = cy_bed + bed_h
            ax.plot([cx - bed_w / 2, cx + bed_w / 2], [sep_y, sep_y],
                    color=METAL_BORDER, lw=1.2, zorder=5, alpha=0.8)

    return bed_cy, cy_bot + total_h




# ══════════════════════════════════════════════════════════════════════
# LÉGENDE FLUIDES
# ══════════════════════════════════════════════════════════════════════

def _legend(ax, x0=0.30, y0=-4.30):
    items = [
        (C_GAZ,       'Gaz process / SO₂ / SO₃'),
        (C_VAP_HP,    'Vapeur HP'),
        (C_VAP_BP,    'Vapeur BP'),
        (C_EAU_ALIM,  'Eau alimentaire / BFW'),
        (C_RECYCLE,   'Recycle SO₂ (gaz froid)'),
        (C_ACIDE_IN,  'Acide H₂SO₄ entrant'),
        (C_ACIDE_OUT, 'Acide H₂SO₄ sortant'),
    ]
    ncols = 2
    col_w = 7.0
    ax.add_patch(FancyBboxPatch((x0 - 0.12, y0 - 0.12),
                               col_w * ncols + 0.24,
                               len(items) // ncols * 0.32 + 0.55,
                               boxstyle="round,pad=0.05",
                               facecolor='#DDDDDD', edgecolor='#888', lw=0.8, zorder=8))
    ax.text(x0 + col_w * ncols / 2, y0 + len(items) // ncols * 0.32 + 0.30,
            'LÉGENDE FLUIDES', color='#111', fontsize=7.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=9)
    for i, (c, lbl) in enumerate(items):
        col = i % ncols
        row = i // ncols
        xi = x0 + col * col_w
        yi = y0 + row * 0.32
        ax.plot([xi + 0.05, xi + 0.55], [yi + 0.13, yi + 0.13],
                color=c, lw=3.5, zorder=10)
        ax.text(xi + 0.65, yi + 0.13, lbl, color='#111', fontsize=6.5,
                va='center', fontfamily='monospace', zorder=10)


# ══════════════════════════════════════════════════════════════════════
# RENDER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

def render_page_exchangers(
    hx_results: dict,
    conv_results: dict | None = None,
) -> plt.Figure:

    if conv_results is None:
        conv_results = {
            'T_in_lits':  [420., 527., 440., 393.],
            'T_out_lits': [627., 516., 460., 406.],
            'tau_lits':   [65.92, 73.25, 64.48, 98.99],
            'dp_lits':    [1.69, 1.75, 2.14, 1.90],
        }

    r_sh   = hx_results.get('hp_superheater_1b', {})
    r_hi   = hx_results.get('hot_interpass',     {})
    r_ci   = hx_results.get('cold_interpass',    {})
    r_ec   = hx_results.get('economizer_3b',     {})
    r_hp4a = hx_results.get('hp4a',  {})
    r_lp4a = hx_results.get('lp4a',  {})
    r_e4c  = hx_results.get('e4c',   {})
    r_e4a  = hx_results.get('e4a',   {})

    # ── Canvas ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(52, 22))
    _unisim_bg(fig, ax, xlim=(0, 52), ylim=(-4.5, 22.0))
    _title_bar(ax,
               'SUPERVISION UNISIM-DESIGN — ÉCHANGEURS DE CHALEUR & CONVERTISSEUR',
               xlim=52, ylim=22.0)

    # ── Positions clés ────────────────────────────────────────────────
    CX    = 22.0     # centre convertisseur
    BW    = 3.80     # largeur lit
    BH    = 2.10     # hauteur lit
    BG_G  = 0.45     # gap entre lits
    CY_B  = 5.50     # bas convertisseur

    bed_cy, cy_top = _converter_unisim(ax, CX, CY_B, BW, BH, BG_G, conv_results)

    cxL = CX - BW / 2   # bord gauche convertisseur
    cxR = CX + BW / 2   # bord droit convertisseur

    X_GL  = cxL - 2.40   # rail gauche gaz
    X_GR  = cxR + 2.40   # rail droit gaz
    X_RC  = cxL - 5.80   # rail recycle (séparé)

    # ── CHAUDIÈRE DE RÉCUPÉRATION → LIT 1 ────────────────────────────
    T_in_b1 = conv_results['T_in_lits'][0]
    ax.add_patch(FancyBboxPatch((0.20, bed_cy[0] - 0.38), 3.20, 0.76,
                               boxstyle="round,pad=0.07", lw=1.2,
                               edgecolor=METAL_BORDER, facecolor=METAL_MID, zorder=5))
    ax.text(1.80, bed_cy[0] + 0.06, 'Chaudière de',
            color='#111', fontsize=7, ha='center', va='center',
            fontfamily='monospace', fontweight='bold', zorder=6)
    ax.text(1.80, bed_cy[0] - 0.18, 'récupération 401AV01',
            color='#333', fontsize=6.5, ha='center', va='center',
            fontfamily='monospace', zorder=6)

    _pipe(ax, 3.40, bed_cy[0], X_GL, bed_cy[0], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GL, bed_cy[0], cxL - 0.01, bed_cy[0], color=C_GAZ, lw=2.5, tag='F-L1IN')

    _data_box(ax, cxL - 2.05, bed_cy[0] + 0.18, [
        (f"{T_in_b1:.0f}", "°C", FG_DATA3),
    ], w=1.80)

    # ── HP SUPERHEATER 1B — LIT1→LIT2, côté DROIT ────────────────────
    SH_W = 1.50
    SH_H = 2.60
    CX_SH = cxR + 5.80
    SH_MID_Y = (bed_cy[0] + bed_cy[1]) / 2
    CY_SH_B  = SH_MID_Y - SH_H / 2

    _superheater_unisim(ax, CX_SH, CY_SH_B, SH_W, SH_H, 'HP\nSuperheater\n1B\n401AE05')

    # LIT1 droite → X_GR → SH
    _pipe(ax, cxR + 0.01, bed_cy[0], X_GR, bed_cy[0], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, bed_cy[0], X_GR, SH_MID_Y, color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, SH_MID_Y, CX_SH - SH_W / 2 - 0.01, SH_MID_Y,
          color=C_GAZ, lw=2.5, tag='F-SH-IN')

    # SH → LIT2
    X_SH_R = CX_SH + SH_W / 2 + 0.70
    _pipe(ax, CX_SH + SH_W / 2 + 0.01, SH_MID_Y, X_SH_R, SH_MID_Y,
          color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_SH_R, SH_MID_Y, X_SH_R, bed_cy[1], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_SH_R, bed_cy[1], cxR + 0.01, bed_cy[1], color=C_GAZ, lw=2.5, tag='F-SH-OUT')

    # Vapeur HP entrée/sortie
    dome_top = CY_SH_B + SH_H + SH_W * 0.35 * 0.7 + 0.20
    _pipe(ax, CX_SH, CY_SH_B - 0.70, CX_SH, CY_SH_B - 0.01,
          color=C_VAP_HP, lw=2.0, tag='F-VAP-IN')
    _pipe(ax, CX_SH, dome_top, CX_SH, dome_top + 0.65,
          color=C_VAP_HP, lw=2.0, tag='F-VAP-OUT')

    # Data SH
    dx_sh = CX_SH + SH_W / 2 + 0.90
    _data_box(ax, dx_sh, CY_SH_B + SH_H * 0.55, [
        (f"{r_sh.get('T_gas_in_C', 627):.0f}", "°C gaz in", FG_DATA3),
        (f"{r_sh.get('T_gas_out_C', 440):.0f}", "°C gaz out", FG_DATA2),
        (f"{r_sh.get('Q_exchanged_MW', 0):.2f}", "MW", FG_DATA),
        (f"{r_sh.get('effectiveness_pct', 0):.1f}", "% η", FG_DATA),
    ], w=2.10)
    _data_box(ax, CX_SH - SH_W / 2 - 2.20, CY_SH_B - 0.20, [
        (f"{r_sh.get('T_cold_in_C', 263):.0f}", "°C vap in", FG_DATA3),
        (f"{r_sh.get('T_cold_out_C', 342):.0f}", "°C vap out", FG_DATA2),
    ], w=2.10)

    # ── HOT INTERPASS HX — LIT2→LIT3, côté GAUCHE ────────────────────
    HX_W = 3.60
    HX_H = 1.30
    CX_HI = cxL - 8.50
    CY_HI = (bed_cy[1] + bed_cy[2]) / 2

    _hx_unisim(ax, CX_HI, CY_HI, HX_W, HX_H,
               'Hot Interpass HX', '401AE19')
    _stream_tag(ax, CX_HI, CY_HI + HX_H / 2 + 0.32, '401AE19')

    cxHI_L = CX_HI - HX_W / 2
    cxHI_R = CX_HI + HX_W / 2
    OFF = 0.32

    # LIT2 gauche → X_GL monte → HX droite haut
    _pipe(ax, cxL - 0.01, bed_cy[1], X_GL, bed_cy[1], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GL, bed_cy[1], X_GL, CY_HI + OFF, color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GL, CY_HI + OFF, cxHI_R + 0.01, CY_HI + OFF,
          color=C_GAZ, lw=2.5, tag='F-HI-IN')

    # HX sortie gauche bas → X_GL descend → LIT3
    _pipe(ax, cxHI_L - 0.01, CY_HI - OFF, X_GL, CY_HI - OFF, color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GL, CY_HI - OFF, X_GL, bed_cy[2], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GL, bed_cy[2], cxL - 0.01, bed_cy[2], color=C_GAZ, lw=2.5, tag='F-HI-OUT')

    # Data HI
    dx_hi = cxHI_L - 2.30
    _data_box(ax, dx_hi, CY_HI + 0.50, [
        (f"{r_hi.get('T_gas_in_C', 516):.0f}", "°C gaz in", FG_DATA3),
        (f"{r_hi.get('T_gas_out_C', 440):.0f}", "°C gaz out", FG_DATA2),
        (f"{r_hi.get('Q_exchanged_MW', 0):.2f}", "MW", FG_DATA),
        (f"{r_hi.get('effectiveness_pct', 0):.1f}", "% η", FG_DATA),
    ], w=2.10)

    # ── COLD INTERPASS HX — LIT3→LIT4, côté DROIT ────────────────────
    CX_CI = cxR + 8.20
    CY_CI = (bed_cy[2] + bed_cy[3]) / 2

    _hx_unisim(ax, CX_CI, CY_CI, HX_W, HX_H,
               'Cold Interpass HX', '401AE18')
    _stream_tag(ax, CX_CI, CY_CI + HX_H / 2 + 0.32, '401AE18')

    cxCI_L = CX_CI - HX_W / 2
    cxCI_R = CX_CI + HX_W / 2
    Y_CI_IN  = CY_CI + OFF
    Y_CI_OUT = CY_CI - OFF

    # LIT3 droite → X_GR → CI gauche haut
    _pipe(ax, cxR + 0.01, bed_cy[2], X_GR, bed_cy[2], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, bed_cy[2], X_GR, Y_CI_IN, color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, Y_CI_IN, cxCI_L - 0.01, Y_CI_IN, color=C_GAZ, lw=2.5, tag='F-CI-IN')

    # Data CI
    dx_ci = cxCI_R + 0.18
    _data_box(ax, dx_ci, CY_CI + 0.50, [
        (f"{r_ci.get('T_gas_in_C', 460):.0f}", "°C gaz in", FG_DATA3),
        (f"{r_ci.get('T_gas_out_C', 306):.0f}", "°C gaz out", FG_DATA2),
        (f"{r_ci.get('Q_exchanged_MW', 0):.2f}", "MW", FG_DATA),
    ], w=2.10)

    # ── ÉCONOMISEUR 3B — suite CI ─────────────────────────────────────
    EC_W = 3.20
    EC_H = 1.15
    CX_EC = cxCI_R + EC_W / 2 + 2.00
    CY_EC = Y_CI_OUT

    _pipe(ax, cxCI_R + 0.01, Y_CI_OUT, CX_EC - EC_W / 2 - 0.01, CY_EC,
          color=C_GAZ, lw=2.5, tag='F-EC-IN')
    _hx_unisim(ax, CX_EC, CY_EC, EC_W, EC_H, 'Économiseur 3B', '401AE10')

    dx_ec = CX_EC + EC_W / 2 + 0.15
    _data_box(ax, dx_ec, CY_EC + 0.26, [
        (f"{r_ec.get('T_gas_in_C', 306):.0f}", "°C gaz in", FG_DATA3),
        (f"{r_ec.get('T_gas_out_C', 180):.0f}", "°C gaz out", FG_DATA2),
        (f"{r_ec.get('Q_exchanged_MW', 0):.2f}", "MW", FG_DATA),
    ], w=2.10)

    # ── JD02 — TOUR D'ABSORPTION INTERMÉDIAIRE ────────────────────────
    TW, TH = 1.05, 1.70
    CX_JD02 = CX_EC + EC_W / 2 + TW / 2 + 1.60
    CY_JD02 = CY_EC

    _pipe(ax, CX_EC + EC_W / 2 + 0.01, CY_EC, CX_JD02 - TW / 2 - 0.01, CY_JD02,
          color=C_GAZ, lw=2.5, tag='F-JD02-IN')
    _tower_unisim(ax, CX_JD02, CY_JD02, TW, TH, 'JD02\nINTER', '401AJ02')

    _pipe(ax, CX_JD02, CY_JD02 + TH / 2 + 0.40, CX_JD02, CY_JD02 + TH / 2,
          color=C_ACIDE_IN, lw=2.0, tag='F-AIN2')
    _pipe(ax, CX_JD02, CY_JD02 - TH / 2, CX_JD02, CY_JD02 - TH / 2 - 0.40,
          color=C_ACIDE_OUT, lw=2.0, tag='F-AOUT2')
    ax.text(CX_JD02, CY_JD02 + TH / 2 + 0.65, "H₂SO₄ IN",
            color=C_ACIDE_IN, fontsize=6.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=10)
    ax.text(CX_JD02, CY_JD02 - TH / 2 - 0.65, "H₂SO₄ OUT",
            color=C_ACIDE_OUT, fontsize=6.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=10)

    # ── BOUCLE RECYCLE SO₂ ────────────────────────────────────────────
    T_r1 = r_ci.get('T_cold_out_C', 200)
    T_r2 = r_hi.get('T_cold_out_C', 425)
    Y_REC  = CY_B - 2.40
    Y_REC2 = Y_REC + 0.60

    X_CI_COLD_IN  = cxCI_R
    X_CI_COLD_OUT = cxCI_L
    X_HI_COLD_IN  = cxHI_R
    X_HI_COLD_OUT = cxHI_L

    # JD02 bas → rail recycle → CI côté utility
    _line(ax, [CX_JD02, CX_JD02], [CY_JD02 - TH / 2, Y_REC], C_RECYCLE, lw=2.2)
    _line(ax, [CX_JD02, X_CI_COLD_IN], [Y_REC, Y_REC], C_RECYCLE, lw=2.2)
    _arr(ax, X_CI_COLD_IN - 0.20, Y_REC, X_CI_COLD_IN, Y_REC, C_RECYCLE, lw=2.2)
    _line(ax, [X_CI_COLD_IN, X_CI_COLD_IN], [Y_REC, CY_CI - HX_H / 2], C_RECYCLE, lw=2.0)
    _arr(ax, X_CI_COLD_IN, CY_CI - HX_H / 2, X_CI_COLD_IN, CY_CI + HX_H / 2, C_RECYCLE, lw=2.0)

    # CI utility out → HI utility in
    _line(ax, [X_CI_COLD_OUT, X_CI_COLD_OUT], [CY_CI + HX_H / 2, Y_REC2], C_RECYCLE, lw=2.0)
    _line(ax, [X_CI_COLD_OUT, X_HI_COLD_IN], [Y_REC2, Y_REC2], C_RECYCLE, lw=2.2)
    _arr(ax, X_HI_COLD_IN - 0.20, Y_REC2, X_HI_COLD_IN, Y_REC2, C_RECYCLE, lw=2.2)
    _line(ax, [X_HI_COLD_IN, X_HI_COLD_IN], [Y_REC2, CY_HI - HX_H / 2], C_RECYCLE, lw=2.0)
    _arr(ax, X_HI_COLD_IN, CY_HI - HX_H / 2, X_HI_COLD_IN, CY_HI + HX_H / 2, C_RECYCLE, lw=2.0)

    # HI utility out → LIT4 entrée gauche
    _line(ax, [X_HI_COLD_OUT, X_HI_COLD_OUT], [CY_HI + HX_H / 2, bed_cy[3]], C_RECYCLE, lw=2.2)
    _arr(ax, X_HI_COLD_OUT, bed_cy[3], cxL - 0.01, bed_cy[3], C_RECYCLE, lw=2.2)

    # Étiquette recycle
    ax.text((CX_JD02 + X_CI_COLD_IN) / 2 - 3.0, Y_REC - 0.22,
            f"SO₂ recyclé  {T_r1:.0f}→{T_r2:.0f}°C",
            color=C_RECYCLE, fontsize=7, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=10)

    # ── SECTION FINALE : LIT4 → HP4A → LP4A → E4C → E4A → JD03 ──────
    FHX_W = 3.50
    FHX_H = 1.35
    FHX_GAP = 1.90
    Y_FIN = CY_B - 4.10

    fhx_data = [
        ('HP 4A', '401AE20', r_hp4a, C_VAP_HP),
        ('LP 4A', '401AE21', r_lp4a, C_VAP_BP),
        ('E 4C',  '401AE22', r_e4c,  C_EAU_ALIM),
        ('E 4A',  '401AE23', r_e4a,  C_EAU_ALIM),
    ]

    TW2, TH2 = 1.10, 1.90
    CX_JD03 = 50.50
    CY_JD03 = Y_FIN

    CX_F0 = X_GR + 1.10 + FHX_W / 2
    fhx_cx = [CX_F0 + k * (FHX_W + FHX_GAP) for k in range(len(fhx_data))]

    for k, (name, tag_eq, r, c_util) in enumerate(fhx_data):
        cx_k = fhx_cx[k]
        _hx_unisim(ax, cx_k, Y_FIN, FHX_W, FHX_H, name, tag_eq)

        # Tags flux
        _stream_tag(ax, cx_k, Y_FIN + FHX_H / 2 + 0.32, f'F{18 + k}')

        # Data au-dessus
        _data_box(ax, cx_k - FHX_W / 2, Y_FIN + FHX_H / 2 + 0.55, [
            (f"{r.get('T_gas_in_C', 0):.0f}", "°C in", FG_DATA3),
            (f"{r.get('T_gas_out_C', 0):.0f}", "°C out", FG_DATA2),
            (f"{r.get('Q_exchanged_MW', 0):.2f}", "MW", FG_DATA),
        ], w=FHX_W)

        # Utilité dessous
        _data_box(ax, cx_k - FHX_W / 2, Y_FIN - FHX_H / 2 - 1.0, [
            (f"{r.get('T_cold_out_C', 0):.0f}", "°C util out", c_util),
        ], w=FHX_W)

    # LIT4 → descend via X_GR → HP4A
    _pipe(ax, cxR + 0.01, bed_cy[3], X_GR, bed_cy[3], color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, bed_cy[3], X_GR, Y_FIN, color=C_GAZ, lw=2.5, arrow=False)
    _pipe(ax, X_GR, Y_FIN, fhx_cx[0] - FHX_W / 2 - 0.01, Y_FIN,
          color=C_GAZ, lw=2.5, tag='F-L4OUT')

    # Connexions HX finaux
    for k in range(len(fhx_cx) - 1):
        x1 = fhx_cx[k] + FHX_W / 2 + 0.01
        x2 = fhx_cx[k + 1] - FHX_W / 2 - 0.01
        T_m = fhx_data[k][2].get('T_gas_out_C', 0)
        _pipe(ax, x1, Y_FIN, x2, Y_FIN, color=C_GAZ, lw=2.5, tag=f'F{19 + k}')
        ax.text((x1 + x2) / 2, Y_FIN + 0.16, f"{T_m:.0f}°C",
                color=FG_DATA3, fontsize=7, fontweight='bold',
                ha='center', fontfamily='monospace', zorder=12)

    # Dernier HX → JD03
    T_in_jd03 = r_e4a.get('T_gas_out_C', 134)
    x_last_r = fhx_cx[-1] + FHX_W / 2
    _pipe(ax, x_last_r + 0.01, Y_FIN, CX_JD03 - TW2 / 2 - 0.01, CY_JD03,
          color=C_GAZ, lw=2.5, tag='F-JD03-IN')

    _tower_unisim(ax, CX_JD03, CY_JD03, TW2, TH2, 'JD03\nFINAL', '401AJ03')

    _pipe(ax, CX_JD03, CY_JD03 + TH2 / 2 + 0.42, CX_JD03, CY_JD03 + TH2 / 2,
          color=C_ACIDE_IN, lw=2.0, tag='F-AIN3')
    _pipe(ax, CX_JD03, CY_JD03 - TH2 / 2, CX_JD03, CY_JD03 - TH2 / 2 - 0.42,
          color=C_ACIDE_OUT, lw=2.0, tag='F-AOUT3')
    ax.text(CX_JD03, CY_JD03 + TH2 / 2 + 0.68, "H₂SO₄ IN",
            color=C_ACIDE_IN, fontsize=6.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=10)
    ax.text(CX_JD03, CY_JD03 - TH2 / 2 - 0.68, "H₂SO₄ OUT",
            color=C_ACIDE_OUT, fontsize=6.5, fontweight='bold',
            ha='center', fontfamily='monospace', zorder=10)

    # Gaz traité → cheminée
    _pipe(ax, CX_JD03, CY_JD03 - TH2 / 2 - 0.65,
          CX_JD03, CY_JD03 - TH2 / 2 - 1.25, color='#4488AA', lw=2.0)
    ax.add_patch(FancyBboxPatch((CX_JD03 - 1.20, CY_JD03 - TH2 / 2 - 1.60),
                               2.40, 0.38,
                               boxstyle="round,pad=0.04",
                               facecolor='#CCDDEE', edgecolor='#446688', lw=1.0, zorder=12))
    ax.text(CX_JD03, CY_JD03 - TH2 / 2 - 1.42,
            "Gaz traité → Cheminée", color='#224466', fontsize=7.5,
            fontweight='bold', ha='center', va='center',
            fontfamily='monospace', zorder=13)

    # ── Data JD03
    _data_box(ax, CX_JD03 - TW2 / 2 - 2.30, CY_JD03 - 0.30, [
        (f"{T_in_jd03:.0f}", "°C gaz in", FG_DATA3),
    ], w=2.10)

    # ── SÉPARATEURS DE SECTION ────────────────────────────────────────
    Y_SEP1 = CY_B - 1.30
    Y_SEP2 = Y_FIN + FHX_H / 2 + 0.90
    for ys in [Y_SEP1, Y_SEP2]:
        ax.plot([0.30, 51.70], [ys, ys], color=METAL_BORDER, lw=0.8,
                linestyle='--', zorder=3, alpha=0.6)

    # Section labels
    ax.text(0.50, Y_SEP1 + 0.20, 'SECTION CONVERTISSEUR / ÉCHANGEURS INTERPASS',
            color='#333', fontsize=7.5, fontweight='bold',
            fontfamily='monospace', zorder=10)
    ax.text(0.50, Y_SEP2 - 0.30, 'SECTION FINALE — REFROIDISSEMENT + ABSORPTION JD03',
            color='#333', fontsize=7.5, fontweight='bold',
            fontfamily='monospace', zorder=10)

    # ── LÉGENDE & BOUTONS ─────────────────────────────────────────────
    _legend(ax, x0=0.30, y0=-4.35)
    

    return fig