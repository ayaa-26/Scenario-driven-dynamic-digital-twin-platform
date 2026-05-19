# =====================================================================
# four.py  –  Jumeau numérique du Four de Combustion de Soufre
#
# Architecture quadrizonale hybride CSTR / PFR
# Résolution numérique : schéma Upwind explicite, condition CFL
# Dynamique globale    : filtre passe-bas du 1er ordre
# Pertes pariétales    : résistances en série multicouche
# Thermodynamique      : polynômes de Shomate, Sutherland, Wilke
# =====================================================================

import numpy as np
from scipy.optimize import fsolve

# =====================================================================
# 1. CONSTANTES PHYSIQUES
# =====================================================================

R_CONST  = 8.314           # J/(mol.K)   — constante universelle des gaz parfaits
P_ABS    = 150_000.0       # Pa          — pression opératoire (H8)
T_REF_C  = 25.0            # °C          — température de référence pour les enthalpies Shomate
T_REF_K  = T_REF_C + 273.15
T_AMB_K  = 298.15          # K           — température ambiante pour la convection externe

# Chaleur de réaction de combustion du soufre :
#   S(l) + O₂(g) → SO₂(g)     ΔH_R = −297 000 J/mol  (exothermique)
DELTA_H_R = -297_000.0     # J/mol

# Masses molaires [kg/mol] de chaque espèce
MOLAR_MASS = {
    'SO2':   0.06407,
    'O2':    0.03200,
    'N2':    0.02801,
    'SO3':   0.08006,
    'S_liq': 0.03206,
}

# ── Coefficients du polynôme de Shomate pour Cp [J/(mol.K)] ──────────
# Modèle :  Cp(T) = a + b·T + c·T² + d·T³ + e·T⁴    (T en °C)
# Source NIST ; coefficients S_liq estimés (données manquantes)
COEFFS = {
    'SO2':   [25.3,  5.31e-2, -3.90e-5,  14.5e-9,  -2.01e-12],
    'O2':    [29.1,  1.15e-2, -0.60e-5,   1.31e-9,  -0.16e-12],
    'N2':    [28.9,  0.15e-2,  0.80e-5,  -2.87e-9,   0.35e-12],
    'S_liq': [22.0,  1.50e-2, -1.00e-5,   3.00e-9,  -3.00e-13],  # estimés
}

# ── Paramètres de Sutherland pour la viscosité dynamique ─────────────
# Modèle (éq. 10) :  μ(T) = μ₀ · (T/T₀)^1.5 · (T₀ + S) / (T + S)
# Triplets : (μ₀ [Pa.s], T₀ [K], S [K])
SUTHERLAND = {
    'O2':  (2.07e-5, 291.15, 127.0),
    'N2':  (1.76e-5, 300.55, 111.0),
    'SO2': (1.20e-5, 293.15, 240.0),
}

# =====================================================================
# 2. GÉOMÉTRIE DU FOUR
# =====================================================================

D_INT = 5.4                # m   diamètre intérieur de la virole
R_INT = D_INT / 2.0        # m   rayon intérieur

# Longueurs des quatre zones [m]
#   Z1 (CSTR combustion primaire) | Z2 (PFR transport I)
#   Z3 (CSTR combustion secondaire) | Z4 (PFR transport II)
L1, L2, L3, L4 = 1.5, 5.0, 3.0, 5.5

# Épaisseurs des couches de la paroi [m]
E_REFRACTAIRE = 0.230      # briques réfractaires (couche intérieure)
E_ISOLANT     = 0.114      # briques isolantes (couche intermédiaire)
E_ACIER       = 0.020      # acier au carbone (couche extérieure)

# Rayons successifs des couches [m]
R_REFRAC = R_INT + E_REFRACTAIRE   # interface réfractaire / isolant
R_ISOL   = R_REFRAC + E_ISOLANT    # interface isolant / acier
R_EXT    = R_ISOL   + E_ACIER      # surface externe (côté air)

# Conductivités thermiques des matériaux [W/(m.K)]
LAMBDA_REFRAC = 1.2
LAMBDA_ISOL   = 0.5
LAMBDA_ACIER  = 45.0

# Coefficient de convection externe (air naturel) [W/(m².K)]
H_EXT = 10.0

# ── Rayonnement interne ───────────────────────────────────────────────
# Modèle de rayonnement linéarisé (H10-H11) :
#   q_rad = ε · σ · (Tg² + Tp²) · (Tg + Tp) · (Tg − Tp)
EPSILON      = 0.35                  # émissivité de la paroi réfractaire
SIGMA        = 5.67e-8               # W/(m².K⁴)  constante de Stefan-Boltzmann
T_PAROI_INT  = 800.0 + 273.15        # K  température paroi quasi-stationnaire (H13)

# Condition initiale des PFR (température de démarrage) [K]
T0_INIT  = 900.0 + 273.15

# Nombre de mailles de discrétisation par zone PFR
N_MAILLES = 20

# Volume molaire dans les conditions standard [m³/mol]
V_MOL_STP = 0.022_414

# =====================================================================
# 3. CONDITIONS OPÉRATOIRES PAR DÉFAUT
# =====================================================================

# Nota : le débit de soufre est en kg/MIN (unité procédé industriel)
DEFAULT_F_AIR_NM3H     = 363_934.0  # Nm³/h
DEFAULT_F_SOUFRE_KGMIN = 955.0      # kg/min
DEFAULT_T_AIR_C        = 123.0      # °C
DEFAULT_T_SOUFRE_C     = 132.0      # °C
DEFAULT_BYPASS_PCT     = 10.0       # % air secondaire (bypass)

# =====================================================================
# 4. FONCTIONS THERMODYNAMIQUES
# =====================================================================

def _h_brut(gas, T_c):
    """Primitive du polynôme de Shomate [J/mol] :
       H(T) = a·T + (b/2)·T² + (c/3)·T³ + (d/4)·T⁴ + (e/5)·T⁵
       Utilisée en différence pour obtenir l'enthalpie sensible.
    """
    a, b, c, d, e = COEFFS[gas]
    return (a*T_c + (b/2)*T_c**2 + (c/3)*T_c**3
            + (d/4)*T_c**4 + (e/5)*T_c**5)


def h_sensible(gas, T_c):
    """Enthalpie sensible molaire [J/mol] entre T_REF_C et T_c [°C] :
       Δh(T) = H(T) − H(T_ref)
       Intégrale de Cp(T) dT depuis la référence (25 °C).
    """
    return _h_brut(gas, T_c) - _h_brut(gas, T_REF_C)


def cp_mol(gas, T_c):
    """Capacité thermique molaire isobare [J/(mol.K)] au polynôme de Shomate :
       Cp(T) = a + b·T + c·T² + d·T³ + e·T⁴     (T en °C)
    """
    a, b, c, d, e = COEFFS[gas]
    return a + b*T_c + c*T_c**2 + d*T_c**3 + e*T_c**4


def H_tot(T_c, flux):
    """Débit enthalpique total [J/s = W] à T_c [°C] :
       Ḣ = Σᵢ ṅᵢ · Δhᵢ(T)
       Somme des produits (débit molaire × enthalpie sensible) pour chaque espèce.
    """
    return sum(n * h_sensible(gas, T_c) for gas, n in flux.items())


# ── Viscosité ─────────────────────────────────────────────────────────

def mu_pur(gas, T_K):
    """Viscosité dynamique d'un gaz pur [Pa.s] — corrélation de Sutherland (éq. 10) :
       μ(T) = μ₀ · (T/T₀)^1.5 · (T₀ + S) / (T + S)
       Valable pour des gaz monoatomiques et polyatomiques (écart < 2 % jusqu'à ~1500 K).
    """
    if gas not in SUTHERLAND:
        return 1.5e-5          # valeur par défaut si espèce non renseignée
    mu0, T0, S = SUTHERLAND[gas]
    return mu0 * (T_K / T0)**1.5 * (T0 + S) / (T_K + S)


def mu_mix_wilke(composition, T_K):
    """Viscosité du mélange gazeux [Pa.s] — règle de mélange de Wilke (éq. 11) :
       μ_mix = Σᵢ [ xᵢ·μᵢ / Σⱼ(xⱼ·φᵢⱼ) ]
       φᵢⱼ = [1 + (μᵢ/μⱼ)^0.5 · (Mⱼ/Mᵢ)^0.25]² / [8·(1 + Mᵢ/Mⱼ)]^0.5
       Tient compte des interactions binaires entre espèces de masses molaires différentes.
    """
    species = [g for g in composition
               if composition[g] > 0 and g in MOLAR_MASS and g != 'S_liq']
    if not species:
        return 2e-5
    n_tot = sum(composition[g] for g in species)
    x  = {g: composition[g] / n_tot for g in species}    # fractions molaires
    M  = {g: MOLAR_MASS[g]          for g in species}    # masses molaires
    mu = {g: mu_pur(g, T_K)         for g in species}    # viscosités pures

    mu_mix = 0.0
    for i in species:
        denom = 0.0
        for j in species:
            # Facteur d'interaction binaire φᵢⱼ (Wilke)
            phi = ((1.0 + (mu[i]/mu[j])**0.5 * (M[j]/M[i])**0.25)**2
                   / (8.0 * (1.0 + M[i]/M[j]))**0.5)
            denom += x[j] * phi
        mu_mix += x[i] * mu[i] / denom
    return mu_mix


# ── Conductivité thermique ────────────────────────────────────────────

def lambda_pur(gas, T_K):
    """Conductivité thermique d'un gaz polyatomique [W/(m.K)] (éq. 12) :
       λ = (μ/M) · (Cp + 1.25·R)
       Relation de Eucken modifiée : relie λ à la viscosité et à Cp.
       Le facteur 1.25·R corrige les degrés de liberté internes.
    """
    if gas == 'S_liq' or gas not in MOLAR_MASS:
        return 0.04            # valeur par défaut
    mu_  = mu_pur(gas, T_K)
    Cp_  = cp_mol(gas, T_K - 273.15)
    return mu_ / MOLAR_MASS[gas] * (Cp_ + 1.25 * R_CONST)


def lambda_mix(composition, T_K):
    """Conductivité thermique du mélange [W/(m.K)] — moyenne molaire pondérée :
       λ_mix = Σᵢ xᵢ · λᵢ
       Approximation de première ordre (suffisante pour gaz à composition proche).
    """
    species = [g for g in composition
               if composition[g] > 0 and g in MOLAR_MASS and g != 'S_liq']
    if not species:
        return 0.04
    n_tot = sum(composition[g] for g in species)
    return sum((composition[g] / n_tot) * lambda_pur(g, T_K) for g in species)


# =====================================================================
# 5. PROPRIÉTÉS DU MÉLANGE GAZEUX
# =====================================================================

def mixture_properties(T_K, composition):
    """Calcule les propriétés de transport du mélange gazeux à T_K [K].

    Retourne : (ρ, cp_kg, μ, λ, M_mix)

    Équations mobilisées :
      ρ = P·M_mix / (R·T)                          — gaz parfait
      M_mix = Σᵢ xᵢ·Mᵢ                             — masse molaire moyenne
      Cp_mol_mix = Σᵢ xᵢ·Cpᵢ(T)                    — capacité calorifique molaire
      cp_kg = Cp_mol_mix / M_mix                    — capacité massique
      μ  → règle de Wilke (voir mu_mix_wilke)
      λ  → moyenne molaire (voir lambda_mix)
    """
    species = [g for g in composition
               if composition[g] > 0 and g in MOLAR_MASS and g != 'S_liq']
    if not species:
        return 1.0, 1100.0, 2e-5, 0.05, 0.030

    n_tot      = sum(composition[g] for g in species)
    # Masse molaire moyenne du mélange [kg/mol]
    M_mix      = sum((composition[g] / n_tot) * MOLAR_MASS[g] for g in species)
    # Densité par la loi des gaz parfaits : ρ = P·M/(R·T)
    rho        = P_ABS * M_mix / (R_CONST * T_K)
    # Capacité calorifique molaire puis massique du mélange
    Cp_mol_mix = sum((composition[g] / n_tot) * cp_mol(g, T_K - 273.15)
                     for g in species)
    cp_kg      = Cp_mol_mix / M_mix

    mu_  = mu_mix_wilke(composition, T_K)
    lam_ = lambda_mix(composition, T_K)

    return rho, cp_kg, mu_, lam_, M_mix


# =====================================================================
# 6. COEFFICIENT GLOBAL DE TRANSFERT THERMIQUE (pertes pariétales)
# =====================================================================

def calcul_U_loss(T_K, composition, u_axial):
    """Coefficient global de pertes [W/(m².K)] rapporté à la surface interne.

    Modèle de résistances en série par mètre de four (§5.5, éq. 23) :

    R_total = R_conv_int + R_refrac + R_isol + R_acier + R_conv_ext

    avec (géométrie cylindrique) :
      R_conv_int = 1 / (h_int · π·D_int)          — convection + rayonnement internes
      R_refrac   = ln(r₂/r₁) / (2π·λ_refrac)      — conduction briques réfractaires
      R_isol     = ln(r₃/r₂) / (2π·λ_isol)        — conduction isolant
      R_acier    = ln(r₄/r₃) / (2π·λ_acier)       — conduction acier
      R_conv_ext = 1 / (h_ext · π·D_ext)           — convection externe

    Coefficient interne :
      h_int = h_conv + h_rad        (éq. 19)

    Convection forcée — Dittus-Boelter (éq. 16) :
      Nu = 0.023 · Re^0.8 · Pr^0.4   (turbulent, Re > 4000)
      h_conv = Nu · λ / D_int

    Rayonnement linéarisé (éq. 17-18) :
      h_rad = ε · σ · (Tg² + Tp²) · (Tg + Tp)
      → linéarisation de q = ε·σ·(Tg⁴ − Tp⁴)

    Coefficient global final :
      U_loss = 1 / (R_total · π·D_int)

    Retourne : U_loss, h_conv, h_rad, Re, Pr, Nu
    """
    rho, cp_kg, mu_, lam_, _ = mixture_properties(T_K, composition)

    # ── Nombres adimensionnels (éq. 13-15) ───────────────────────────
    # Reynolds : Re = ρ·u·D / μ   (rapport inertie / viscosité)
    Re = rho * u_axial * D_INT / mu_ if mu_ > 0 else 0.0
    # Prandtl  : Pr = μ·Cp / λ    (rapport diffusivité de quantité de mouvement / thermique)
    Pr = mu_ * cp_kg / lam_ if lam_ > 0 else 0.7

    # ── Corrélation de Nusselt — Dittus-Boelter (éq. 16) ─────────────
    # Régime turbulent  (Re > 4000) : Nu = 0.023·Re^0.8·Pr^0.4
    # Régime de transition interpolé linéairement entre 2300 et 4000
    # Régime laminaire  (Re < 2300) : Nu = 3.66  (conduit isotherme)
    if Re > 4000:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    elif Re > 2300:
        Nu_lam  = 3.66
        Nu_turb = 0.023 * 4000**0.8 * Pr**0.4
        Nu = Nu_lam + (Nu_turb - Nu_lam) * (Re - 2300) / (4000 - 2300)
    else:
        Nu = 3.66

    # Coefficient convectif interne : h_conv = Nu·λ/D
    h_conv = Nu * lam_ / D_INT

    # ── Rayonnement paroi-gaz linéarisé (éq. 17-18) ──────────────────
    # q_rad = ε·σ·(Tg⁴ − Tp⁴) ≈ h_rad·(Tg − Tp)
    # h_rad = ε·σ·(Tg² + Tp²)·(Tg + Tp)
    Tg    = T_K
    Tp    = T_PAROI_INT
    h_rad = EPSILON * SIGMA * (Tg**2 + Tp**2) * (Tg + Tp)

    # Coefficient global interne (éq. 19) : h_int = h_conv + h_rad
    h_int = h_conv + h_rad

    # ── Résistances thermiques par mètre de four (éq. 20-23) ─────────
    A_i = np.pi * D_INT          # surface interne par mètre linéaire [m²/m]
    A_o = np.pi * 2 * R_EXT      # surface externe par mètre linéaire [m²/m]

    # Convection interne : R = 1 / (h · A_i)
    R_conv_i   = 1.0 / (h_int * A_i)  if h_int > 0 else 1e6
    # Conduction cylindrique : R = ln(r_ext/r_int) / (2π·λ)
    R_refrac   = np.log(R_REFRAC / R_INT)    / (2 * np.pi * LAMBDA_REFRAC)
    R_isol     = np.log(R_ISOL   / R_REFRAC) / (2 * np.pi * LAMBDA_ISOL)
    R_acier    = np.log(R_EXT   / R_ISOL)   / (2 * np.pi * LAMBDA_ACIER)
    # Convection externe naturelle : R = 1 / (h_ext · A_o)
    R_conv_ext = 1.0 / (H_EXT * A_o)

    # Résistance totale en série et coefficient global
    R_total = R_conv_i + R_refrac + R_isol + R_acier + R_conv_ext
    # U_loss = 1 / (R_total · A_i)  →  flux par unité de surface interne et d'écart de T
    U_loss  = 1.0 / (R_total * A_i) if R_total > 0 else 0.0

    return U_loss, h_conv, h_rad, Re, Pr, Nu


# =====================================================================
# 7. BILAN MATIÈRE ET TEMPÉRATURES CSTR (régime permanent)
# =====================================================================

def solve_zones(F_air_nm3h, F_soufre_kgmin, bypass_pct,
                T_air_c=DEFAULT_T_AIR_C, T_s_c=DEFAULT_T_SOUFRE_C):
    """Calcule températures et compositions en régime permanent
    par bilan enthalpique sur chaque réacteur CSTR.

    Principe (1er principe appliqué à un réacteur ouvert, régime permanent) :
       Ḣ_sortie = Ḣ_entrée − ξ · ΔH_R      (ΔH_R < 0 : dégagement de chaleur)

    ─── Conversions des débits ───────────────────────────────────────
      ṅ_S    = (F_soufre [kg/min] / 60) · 1000 / 32.06   [mol/s]
      ṅ_air  = (F_air [Nm³/h] / 3600) / V_mol_STP         [mol/s]
      ṅ_air_p = ṅ_air · (1 − bypass/100)   air primaire
      ṅ_air_s = ṅ_air · (bypass/100)        air secondaire

    ─── Zone 1 : mélange (T1) ────────────────────────────────────────
      Bilan enthalpique sans réaction :
        Ḣ_in = Ḣ_air(T_air) + Ḣ_S(T_s)
      T1 résout :  Ḣ({O₂,N₂,S_liq}, T1) = Ḣ_in

    ─── Zone 1 : combustion primaire adiabatique (T_flamme) ──────────
      Avancement : ξ_p = min(1, ṅ_O₂_p / ṅ_S)
      Bilan :  Ḣ_produits = Ḣ_in − ξ_p·ṅ_S·ΔH_R
      T_flamme résout :  Ḣ({SO₂,O₂,N₂}, T_flamme) = Ḣ_produits

    ─── Zone 3 : combustion secondaire + air bypass (T3) ─────────────
      ξ_s = min(1, ṅ_O₂_s / ṅ_S_rest)
      H3 = Ḣ_Z1 + Ḣ_air_s(T_air) − ξ_s·ṅ_S_rest·ΔH_R
      T3 résout :  Ḣ({SO₂,O₂,N₂}_final, T3) = H3

    Retourne un dictionnaire : T1, T_flamme, T3, flux, y_SO2, y_O2, y_SO3, ...
    """
    # ── Débits molaires des réactifs ──────────────────────────────────
    # Soufre liquide : conversion kg/min → mol/s
    n_S       = (F_soufre_kgmin / 60.0) * 1000.0 / 32.06
    # Air total : conversion Nm³/h → mol/s via volume molaire STP
    n_air_tot = (F_air_nm3h / 3600.0) / V_MOL_STP
    # Partitionnement primaire / secondaire selon le taux de bypass
    n_air_p   = n_air_tot * (1.0 - bypass_pct / 100.0)
    n_air_s   = n_air_tot * (bypass_pct / 100.0)

    # Composition de l'air (21 % O₂, 79 % N₂ vol)
    nO2_p = 0.21 * n_air_p
    nN2_p = 0.79 * n_air_p
    nO2_s = 0.21 * n_air_s
    nN2_s = 0.79 * n_air_s

    # ── Zone 1 : température de mélange T1 (avant combustion) ─────────
    # Ḣ_in = Ḣ_air(T_air) + Ḣ_S(T_s) = Σᵢ ṅᵢ·Δhᵢ(T)
    H_in  = (H_tot(T_air_c, {'O2': nO2_p, 'N2': nN2_p})
             + H_tot(T_s_c,  {'S_liq': n_S}))

    flux1 = {'O2': nO2_p, 'N2': nN2_p, 'S_liq': n_S}
    # T1 est la racine de :  Ḣ({O₂,N₂,S_liq}, T1) − Ḣ_in = 0
    T1    = fsolve(lambda T: H_tot(T[0], flux1) - H_in, [T_air_c])[0]

    # ── Zone 1 : combustion primaire adiabatique ───────────────────────
    # Avancement maximal limité par l'O₂ disponible
    xi_p     = min(1.0, nO2_p / max(n_S, 1e-12))
    n_SO2_p  = xi_p * n_S          # moles de SO₂ produites
    n_S_rest = (1.0 - xi_p) * n_S  # soufre non brûlé (envoyé en zone 3)

    # Composition sortie Z1 (gaz uniquement)
    flux2 = {
        'SO2': n_SO2_p,
        'O2':  max(nO2_p - n_SO2_p, 0.0),
        'N2':  nN2_p,
    }
    # Bilan énergétique adiabatique :
    #   Ḣ_produits = Ḣ_in − ξ_p·ṅ_S·ΔH_R   (ΔH_R < 0 → hausse d'enthalpie)
    H2       = H_in - xi_p * n_S * DELTA_H_R
    # T_flamme résout :  Ḣ({SO₂,O₂,N₂}, T_flamme) = H2
    T_flamme = fsolve(lambda T: H_tot(T[0], flux2) - H2, [1176.0])[0]

    # ── Zone 3 : combustion secondaire + mélange air bypass ───────────
    # Combustion du soufre résiduel avec l'O₂ de l'air secondaire
    xi_s    = min(1.0, nO2_s / max(n_S_rest, 1e-12))
    n_SO2_s = xi_s * n_S_rest

    # Composition après mélange et combustion en zone 3
    flux3 = {
        'SO2': n_SO2_p + n_SO2_s,
        'O2':  max(flux2['O2'] + nO2_s - xi_s * n_S_rest, 0.0),
        'N2':  nN2_p + nN2_s,
    }
    # Bilan enthalpique zone 3 :
    #   H3 = Ḣ_Z1 + Ḣ_air_s(T_air) − ξ_s·ṅ_S_rest·ΔH_R
    H3 = (H2
          + H_tot(T_air_c, {'O2': nO2_s, 'N2': nN2_s})
          - xi_s * n_S_rest * DELTA_H_R)
    # T3 résout :  Ḣ({SO₂,O₂,N₂}_final, T3) = H3
    T3 = fsolve(lambda T: H_tot(T[0], flux3) - H3, [1100.0])[0]

    # ── Fractions molaires en sortie ───────────────────────────────────
    F_total = sum(flux3.values())
    y_SO3   = 0.0015                              # 0.15 %mol — valeur empirique
    y_SO2   = flux3['SO2'] / F_total * (1.0 - y_SO3)
    y_O2    = flux3['O2']  / F_total * (1.0 - y_SO3)

    return {
        'T1':       T1,
        'T_flamme': T_flamme,
        'T3':       T3,
        'flux1':    flux1,
        'flux2':    flux2,
        'flux3':    flux3,
        'n_S':      n_S,
        'n_S_rest': n_S_rest,
        'F_total':  F_total,
        'y_SO2':    y_SO2 * 100.0,
        'y_O2':     y_O2  * 100.0,
        'y_SO3':    y_SO3 * 100.0,
        'composition': {
            'SO2': flux3['SO2'],
            'O2':  flux3['O2'],
            'N2':  flux3['N2'],
        },
    }


# =====================================================================
# 8. CLASSE PFRZone  –  Discrétisation Upwind
# =====================================================================

class PFRZone:
    """Zone PFR résolue par un schéma Upwind explicite avec contrôle CFL.

    Équation de transport 1D (advection + perte pariétale) :
       ∂T/∂t + u · ∂T/∂z = −β · (T − T_amb)

    avec :
       u  : vitesse axiale des gaz [m/s]
       β  = 4·U_loss / (D·ρ·cp)   coefficient de perte pariétale [1/s]

    Schéma numérique Upwind (différences finies amont) :
       T_j^{n+1} = T_j^n − CFL·(T_j^n − T_{j-1}^n)   (sans pertes)
       → avec pertes (Euler implicite sur terme source) :
         T_j^{n+1} = (T_adv + dt·β·T_amb) / (1 + dt·β)

    Condition CFL (stabilité) :
       dt_max = 0.8 · dz / u     → CFL ≤ 0.8 (marge de sécurité)
    """

    def __init__(self, name, L, N=N_MAILLES):
        self.name = name
        self.L    = L
        self.N    = N
        self.dz   = L / N                         # pas spatial [m]
        self.z    = np.linspace(0.0, L, N + 1)    # abscisses des nœuds [m]

        # Profil de température initial [K] (uniforme = T0_INIT)
        self.T = np.full(N + 1, T0_INIT)
        self.T_inlet     = T0_INIT
        self.F_total     = 1.0
        self.composition = {'SO2': 60.0, 'O2': 40.0, 'N2': 300.0}

    def set_inlet(self, T_K, F_total, composition):
        """Fixe les conditions d'entrée (condition aux limites amont)."""
        self.T_inlet     = float(T_K)
        self.F_total     = float(F_total)
        self.composition = composition

    def get_outlet(self):
        """Retourne la température en sortie (dernier nœud) [K]."""
        return float(self.T[-1])

    def _vitesse_axiale(self, T_K):
        """Vitesse axiale des gaz [m/s] par la loi des gaz parfaits :
           u = Q / A = (ṅ·R·T/P) / (π·(D/2)²)
        """
        A = np.pi * (D_INT / 2.0)**2          # section droite [m²]
        Q = self.F_total * R_CONST * T_K / P_ABS  # débit volumique [m³/s]
        return max(Q / A, 1e-6)               # garde-fou anti-division par zéro

    def step(self, dt, avec_pertes=True):
        """Avance d'un pas dt [s] avec subdivision automatique en sous-pas CFL.

        Algorithme :
          1. Calcul de u = vitesse axiale à T_entrée
          2. Sous-pas CFL : dt_eff = min(0.8·dz/u, t_restant)
          3. Pour chaque nœud j = 1..N :
             a. T_adv = T_j − CFL·(T_j − T_{j-1})        (advection Upwind)
             b. T_new[j] = (T_adv + dt_eff·β·T_amb) / (1 + dt_eff·β)  (pertes)
          4. Condition de bord aval : Neumann (∂T/∂z = 0)
          5. Répéter jusqu'à épuisement de dt.
        """
        t_restant = dt
        dernier_diag = None

        while t_restant > 1e-9:
            T_old = self.T.copy()
            T_old[0] = self.T_inlet       # condition amont (Dirichlet)

            T_repr = float(T_old[0])
            u = self._vitesse_axiale(T_repr)

            # ── Condition CFL : dt_max = 0.8·dz/u ────────────────────
            dt_max = 0.8 * self.dz / u if u > 0 else dt
            dt_eff = min(dt_max, t_restant)
            cfl_eff = u * dt_eff / self.dz   # nombre de Courant effectif

            # ── Coefficient de perte pariétale ────────────────────────
            # β = 4·U_loss / (D·ρ·cp)    [1/s]
            # Justification : flux de chaleur perdu = U·(T−T_amb)·(π·D·dz)
            #                 terme source volumique = U·4/(D·ρ·cp)·(T−T_amb)
            if avec_pertes:
                U_loss, h_conv, h_rad, Re, Pr, Nu = calcul_U_loss(
                    T_repr, self.composition, u)
                rho, cp_kg, _, _, _ = mixture_properties(T_repr, self.composition)
                beta = (4.0 * U_loss / (D_INT * rho * cp_kg)
                        if rho * cp_kg > 0 else 0.0)
            else:
                U_loss = h_conv = h_rad = Re = Pr = Nu = beta = 0.0

            T_new = T_old.copy()
            for j in range(1, self.N + 1):
                # Étape d'advection Upwind :
                #   T_adv = T_j − CFL·(T_j − T_{j-1})
                T_adv = T_old[j] - cfl_eff * (T_old[j] - T_old[j-1])
                if avec_pertes:
                    # Euler implicite sur le terme de perte pariétale :
                    #   T_new = (T_adv + dt·β·T_amb) / (1 + dt·β)
                    T_new[j] = (T_adv + dt_eff * beta * T_AMB_K) / (1.0 + dt_eff * beta)
                else:
                    T_new[j] = T_adv   # advection pure (adiabatique)

            # Condition de bord aval (Neumann homogène : flux nul en sortie)
            T_new[-1] = T_new[-2]
            self.T = T_new

            dernier_diag = {
                'U_loss': U_loss, 'h_conv': h_conv, 'h_rad': h_rad,
                'Re': Re, 'Pr': Pr, 'Nu': Nu,
                'cfl': cfl_eff, 'u': u,
            }
            t_restant -= dt_eff

        return dernier_diag if dernier_diag else {
            'U_loss': 0, 'h_conv': 0, 'h_rad': 0,
            'Re': 0, 'Pr': 0, 'Nu': 0, 'cfl': 0, 'u': 0,
        }


# =====================================================================
# 9. CLASSE Furnace  –  Orchestration quadrizonale
# =====================================================================

class Furnace:
    """Jumeau numérique du four de combustion de soufre.

    Architecture quadrizonale :
      Z1 (CSTR)  : combustion primaire  — bilan enthalpique adiabatique
      Z2 (PFR)   : transport thermique  — advection + pertes pariétales
      Z3 (CSTR)  : combustion secondaire + mélange air bypass
      Z4 (PFR)   : transport thermique  — advection + pertes pariétales

    Dynamique sortie — filtre passe-bas du 1er ordre (éq. 28, §6.6) :
       y[n+1] = y[n] + α·(y_cible − y[n])
       α = 1 − exp(−dt/τ)      (discrétisation exacte)
       τ_T = 8 s  (inertie thermique),  τ_c = 5 s  (inertie chimique)

    Paramètres
    ----------
    F_air_nm3h     : débit d'air          [Nm³/h]
    F_soufre_kgmin : débit de soufre      [kg/min]
    bypass_pct     : % air secondaire     [%]
    T_air_c        : température air      [°C]
    T_soufre_c     : température soufre   [°C]
    pertes_actives : activer pertes paroi [bool]
    """

    def __init__(self,
                 F_air_nm3h=DEFAULT_F_AIR_NM3H,
                 F_soufre_kgmin=DEFAULT_F_SOUFRE_KGMIN,
                 bypass_pct=DEFAULT_BYPASS_PCT,
                 T_air_c=DEFAULT_T_AIR_C,
                 T_soufre_c=DEFAULT_T_SOUFRE_C,
                 pertes_actives=True):

        self.F_air_nm3h     = F_air_nm3h
        self.F_soufre_kgmin = F_soufre_kgmin
        self.bypass_pct     = bypass_pct
        self.T_air_c        = T_air_c
        self.T_soufre_c     = T_soufre_c
        self.pertes_actives = pertes_actives

        # Instanciation des deux zones PFR avec leur longueur respective
        self.z2 = PFRZone('Z2', L2, N_MAILLES)
        self.z4 = PFRZone('Z4', L4, N_MAILLES)

        # État dynamique filtré — initialisé à T0_INIT
        T0_c = T0_INIT - 273.15
        self.T_out   = T0_c   # température sortie filtrée [°C]
        self.SO2_out = 12.0   # concentration SO₂ filtrée [%vol]
        self.O2_out  =  8.0   # concentration O₂ filtrée  [%vol]
        self.SO3_out =  0.15  # concentration SO₃ filtrée [%vol]

        # Calcul du point d'équilibre (cibles régime permanent)
        self._targets = self._calcul_cibles()

    # ── Calcul des cibles régime permanent ────────────────────────────

    def _calcul_cibles(self):
        """Appelle solve_zones pour calculer l'état stationnaire cible
        correspondant aux paramètres opératoires actuels."""
        return solve_zones(
            self.F_air_nm3h, self.F_soufre_kgmin, self.bypass_pct,
            self.T_air_c, self.T_soufre_c)

    def update_parametres(self, **kwargs):
        """Met à jour les paramètres opératoires et recalcule les cibles
        (simule un changement de consigne ou de débit)."""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._targets = self._calcul_cibles()

    # ── Pas de temps dynamique ─────────────────────────────────────────

    def step(self, dt):
        """Avance la simulation d'un pas dt [s].

        Séquence de calcul (§7.2, Listing 2) :
          1. Z1 (CSTR) : température flamme issue du bilan permanent
          2. Z2 (PFR)  : propagation du profil thermique sur dt
          3. Z3 (CSTR) : bilan enthalpique rigoureux avec air secondaire
          4. Z4 (PFR)  : propagation du profil thermique sur dt
          5. Filtre    : lissage exponentiel vers les cibles

        Zone 3 — bilan détaillé :
          H_in_Z3  = Ḣ_gaz_sortie_Z2(T_Z2) + Ḣ_air_s(T_air)
          H_out_Z3 = H_in_Z3 − ξ_s·ṅ_S_rest·ΔH_R
          T_Z3 résout :  Ḣ(composition_Z3, T_Z3) = H_out_Z3

        Filtre passe-bas (éq. 28) :
          α   = 1 − exp(−dt/τ_T)     avec τ_T = 8 s
          α_c = 1 − exp(−dt/τ_c)     avec τ_c = 5 s
          X_out += α · (X_cible − X_out)
        """
        tgt = self._targets

        # ── Zone 1 : combustion primaire (CSTR) ───────────────────────
        # T_flamme calculée par bilan enthalpique adiabatique dans solve_zones
        T_flamme_K = tgt['T_flamme'] + 273.15

        # ── Zone 2 : transport thermique I (PFR) ──────────────────────
        # Propagation du profil thermique depuis T_flamme vers la sortie Z2
        self.z2.set_inlet(T_flamme_K, tgt['F_total'], tgt['composition'])
        diag_z2    = self.z2.step(dt, avec_pertes=self.pertes_actives)
        T_apres_z2 = self.z2.get_outlet()          # [K]

        # ── Zone 3 : combustion secondaire + air bypass (CSTR) ────────
        # Recalcul des débits d'air secondaire à partir des paramètres courants
        n_air_tot = (self.F_air_nm3h / 3600.0) / V_MOL_STP
        n_air_s   = n_air_tot * (self.bypass_pct / 100.0)
        n_O2_s    = 0.21 * n_air_s
        n_N2_s    = 0.79 * n_air_s

        # Soufre résiduel non brûlé en zone 1
        n_air_p   = n_air_tot * (1.0 - self.bypass_pct / 100.0)
        n_O2_p    = 0.21 * n_air_p
        xi_p      = min(1.0, n_O2_p / max(tgt['n_S'], 1e-12))
        n_S_rest  = (1.0 - xi_p) * tgt['n_S']

        # Avancement de la combustion secondaire (limité par O₂ disponible)
        xi_s      = min(1.0, n_O2_s / max(n_S_rest, 1e-12))
        n_S_brule = xi_s * n_S_rest   # mol/s de S brûlé en zone 3

        # Bilan enthalpique entrée zone 3 :
        #   H_in_Z3 = Ḣ_gaz_Z2(T_Z2_out) + Ḣ_air_secondaire(T_air)
        composition_z2 = self.z2.composition.copy()
        H_in_z3 = (H_tot(T_apres_z2 - 273.15, composition_z2) +
                   H_tot(self.T_air_c, {'O2': n_O2_s, 'N2': n_N2_s}))

        # Enthalpie après combustion secondaire :
        #   H_out_Z3 = H_in_Z3 − ξ_s·ṅ_S_rest·ΔH_R    (ΔH_R < 0)
        H_out_z3 = H_in_z3 - n_S_brule * DELTA_H_R

        # Mise à jour de la composition après mélange et combustion
        composition_z3 = composition_z2.copy()
        composition_z3['SO2'] = composition_z3.get('SO2', 0.0) + n_S_brule
        composition_z3['O2']  = max(composition_z3.get('O2', 0.0) - n_S_brule, 0.0)
        composition_z3['N2']  = composition_z3.get('N2', 0.0) + n_N2_s
        if 'S_liq' in composition_z3:
            del composition_z3['S_liq']   # plus de soufre liquide dans la phase gazeuse

        # Résolution de T_Z3 par équilibre enthalpique adiabatique :
        #   Ḣ(composition_Z3, T_Z3) = H_out_Z3
        def func(T_c):
            return H_tot(T_c, composition_z3) - H_out_z3
        T_apres_z3 = fsolve(func, T_apres_z2 - 273.15)[0] + 273.15

        # ── Zone 4 : transport thermique II (PFR) ─────────────────────
        # Propagation du profil thermique depuis T_Z3 vers la sortie finale
        self.z4.set_inlet(T_apres_z3, tgt['F_total'], composition_z3)
        diag_z4    = self.z4.step(dt, avec_pertes=self.pertes_actives)
        T_sortie_K = self.z4.get_outlet()          # [K]

        # ── Filtre passe-bas du 1er ordre (éq. 28, §6.6) ──────────────
        # Discrétisation exacte : α = 1 − exp(−dt/τ)
        # Simule l'inertie thermique et chimique du four
        tau_T   = 8.0    # s  — constante de temps thermique
        tau_c   = 5.0    # s  — constante de temps chimique
        alpha   = 1.0 - np.exp(-dt / tau_T)    # facteur de lissage thermique
        alpha_c = 1.0 - np.exp(-dt / tau_c)    # facteur de lissage chimique

        # Mise à jour filtrée : X_out += α·(X_cible − X_out)
        self.T_out   += alpha   * ((T_sortie_K - 273.15) - self.T_out)
        self.SO2_out += alpha_c * (tgt['y_SO2'] - self.SO2_out)
        self.O2_out  += alpha_c * (tgt['y_O2']  - self.O2_out)
        self.SO3_out += alpha_c * (tgt['y_SO3'] - self.SO3_out)

        return {
            # Températures par zone [°C]
            'T_flamme':    tgt['T_flamme'],
            'T_z2_sortie': T_apres_z2 - 273.15,
            'T_z3_sortie': T_apres_z3 - 273.15,
            'T_sortie':    self.T_out,        # filtrée [°C]
            # Concentrations filtrées [%vol]
            'SO2': self.SO2_out,
            'O2':  self.O2_out,
            'SO3': self.SO3_out,
            # Profils axiaux PFR [°C]
            'profil_z2': self.z2.T - 273.15,
            'profil_z4': self.z4.T - 273.15,
            'z_z2':      self.z2.z,
            'z_z4':      self.z4.z,
            # Diagnostics transfert thermique
            'diag_z2': diag_z2,
            'diag_z4': diag_z4,
            # Paramètres opératoires courants
            'F_air_nm3h':     self.F_air_nm3h,
            'F_soufre_kgmin': self.F_soufre_kgmin,
            'bypass_pct':     self.bypass_pct,
        }

    # ── Simulation complète ────────────────────────────────────────────

    def run(self, t_total, dt=1.0, callback=None):
        """Simule t_total secondes avec un pas dt [s].

        À chaque pas dt, appelle self.step(dt) et accumule les états.
        callback(t, state) est appelé à chaque pas si fourni
        (utile pour la visualisation temps réel ou la sauvegarde).
        Retourne la liste complète des états (historique).
        """
        historique = []
        t = 0.0
        while t <= t_total + 1e-9:
            s = self.step(dt)
            s['t'] = t
            historique.append(s)
            if callback:
                callback(t, s)
            t += dt
        return historique


# =====================================================================
# 10. POINT D'ENTRÉE – démonstration
# =====================================================================

if __name__ == '__main__':
    SEP = "=" * 65

    print(SEP)
    print("  Jumeau Numérique – Four de Combustion de Soufre")
    print("  Architecture quadrizonale CSTR/PFR ")
    print(SEP)

    # Instanciation aux conditions nominales
    four = Furnace(
        F_air_nm3h     = 363_934.0,
        F_soufre_kgmin = 955.0,
        bypass_pct     = 10.0,
        T_air_c        = 123.0,
        T_soufre_c     = 132.0,
        pertes_actives = True,
    )

    # Affichage du régime permanent calculé par solve_zones
    tgt = four._targets
    print(f"\n── Régime permanent  ──────────────────────────────")
    print(f"  n_S              = {tgt['n_S']:.2f} mol/s")
    print(f"  T1 (avant comb.) = {tgt['T1']:.1f} degC")
    print(f"  T_flamme  (Z1)   = {tgt['T_flamme']:.1f} degC")
    print(f"  T_sortie  (Z3)   = {tgt['T3']:.1f} degC")
    print(f"  SO2 sortie       = {tgt['y_SO2']:.2f} %vol")
    print(f"  O2  sortie       = {tgt['y_O2']:.2f} %vol")
    print(f"  SO3 sortie       = {tgt['y_SO3']:.3f} %vol")
    print(f"  F_total          = {tgt['F_total']:.1f} mol/s")

    # ── Sortie four / Entrée chaudière ────────────────────────────────
    print("\n" + "="*65)
    print("  SORTIE FOUR – ENTRÉE CHAUDIÈRE")
    print("="*65)

    T_gaz_sortie_degC = tgt['T3']
    debit_molaire_total_mol_s = tgt['F_total']

    # Fractions molaires à partir des %vol
    y_SO2 = tgt['y_SO2'] / 100.0
    y_O2  = tgt['y_O2']  / 100.0
    y_SO3 = tgt['y_SO3'] / 100.0
    y_N2  = 1.0 - y_SO2 - y_O2 - y_SO3

    # Masse molaire moyenne du mélange [kg/mol]
    M_mix = (y_SO2 * MOLAR_MASS['SO2'] +
             y_O2  * MOLAR_MASS['O2']  +
             y_SO3 * MOLAR_MASS['SO3'] +
             y_N2  * MOLAR_MASS['N2'])

    # Débit massique total [kg/s] = ṅ_total · M_mix
    debit_massique_kg_s = debit_molaire_total_mol_s * M_mix
    T_K = T_gaz_sortie_degC + 273.15
    composition_pour_cp = tgt['composition']
    _, cp_kg, _, _, _ = mixture_properties(T_K, composition_pour_cp)

    print(f"Température des gaz    : {T_gaz_sortie_degC:.2f} °C")
    print(f"Débit molaire total    : {debit_molaire_total_mol_s:.2f} mol/s")
    print(f"Débit massique total   : {debit_massique_kg_s:.2f} kg/s")
    print(f"Masse molaire moyenne  : {M_mix*1000:.2f} g/mol")
    print(f"Chaleur spécifique     : {cp_kg:.2f} J/(kg·K)")
    print(f"Composition (% vol)    : SO2={y_SO2*100:.3f}% O2={y_O2*100:.3f}% "
          f"SO3={y_SO3*100:.3f}% N2={y_N2*100:.3f}%")
    print("="*65)

    # ── Simulation dynamique 60 s ─────────────────────────────────────
    print(f"\n── Simulation dynamique – démarrage (60 s, dt=1 s) ────────")
    print(f"{'t [s]':>6}  {'T_sortie':>10}")
    print("-" * 70)

    for t in range(0, 65, 5):
        s = four.step(dt=1.0)
        print(f"{t:>6}  {s['T_sortie']:>10.1f}")

    # ── Diagnostic transfert thermique zone 2 ─────────────────────────
    print(f"\n── Transfert thermique Zone 2  ───────────────")
    d = s['diag_z2']
    if d:
        print(f"  Re      = {d['Re']:.0f}")
        print(f"  Pr      = {d['Pr']:.3f}")
        print(f"  Nu      = {d['Nu']:.1f}")
        print(f"  h_conv  = {d['h_conv']:.2f} W/(m2.K)")
        print(f"  h_rad   = {d['h_rad']:.2f} W/(m2.K)")
        print(f"  U_loss  = {d['U_loss']:.4f} W/(m2.K)")
        print(f"  u_gaz   = {d['u']:.3f} m/s")

    print(f"\n  Simulation terminée.")
     
    