# =====================================================================
# simulation/main_simulation.py
# Orchestre : Four (Furnace) → Chaudière → Conv (lits 1-3) → JD02 → Conv (lit 4) → JD03
#
# Version hybride :
#   - Four : dynamique (classe Furnace)
#   - Convertisseur : statique (intégration spatiale odeint)
#
# Ajouts v3 (page DYNAMIQUE) :
#   - Profils spatiaux complets par lit (z, T, tau, r) exportés
#   - Historique temporel simulé (réponse à un échelon de débit)
#   - Données chaudière enrichies (profil T, efficacité)
#   - Données four enrichies (profil T flamme, rendement combustion)
# =====================================================================

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy.integrate import odeint

from model.four import (
    Furnace,
    R_CONST, P_ABS,
)
from model.chaudiere     import solve_boiler_and_bypass, T_TARGET_CONV
from model.absorption    import AbsorptionTowerInter, AbsorptionTowerFinal


# ═════════════════════════════════════════════════════════════════════
# Classe statique du convertisseur
# ═════════════════════════════════════════════════════════════════════

class ConvertisseurStatique:
    """Version statique (intégration spatiale par odeint)."""

    def __init__(self):
        self.R        = 8.314
        self.deltaH_r = -99000.0
        self.rho_cat  = 550.0
        self.epsilon  = 0.523
        self.Dr       = 12.8
        self.A_lit    = 128.68

        self.V_cat_lits  = [87.0, 103.0, 134.0, 167.0]
        self.H_cat_reel  = [V / self.A_lit for V in self.V_cat_lits]
        self.n_lits      = len(self.H_cat_reel)

        self.SHOMATE = {
            'N2':  [19.50583,  19.88705,  -8.598535,  1.369784,  0.527601],
            'O2':  [30.03235,   8.772972, -3.988133,  0.788313, -0.741599],
            'SO2': [21.43049,  74.35094, -57.75217,  16.35534,  0.086731],
            'SO3': [24.02503, 119.4607,  -94.38686,  26.96237, -0.117517],
        }
        self.MOLAR_MASS = {
            'SO2': 0.06407, 'O2': 0.03200,
            'N2':  0.02801, 'SO3': 0.08006
        }
        self.cinetique = {
            'K1': {'A': 2.15e13, 'E': 98900.0},
            'K3': {'A': 7.8e-2,  'E':  6280.0},
            'K4': {'A': 1.3e4,   'E': 25100.0},
        }
        self.facteur_activite_lits = [4500.0, 11000.0, 18000.0, 45000.0]

        self.Q_in   = 624600.0 / 3600.0
        self.P_in   = 150000.0
        self.y0     = {
            'SO2': 0.1101, 'O2': 0.0999,
            'SO3': 0.000165,
            'N2':  1.0 - (0.1101 + 0.0999 + 0.000165)
        }
        self.T_in_lits_C = [420.0, 454.0, 449.0, 425.0]
        self.D_p    = 0.005
        self.mu_ref = {'SO2': 1.23e-5, 'O2': 1.92e-5, 'N2': 1.66e-5, 'SO3': 1.45e-5}

    def set_conditions(self, Q_in, P_in, y0, T_entree):
        self.Q_in        = Q_in / 3600.0
        self.P_in        = P_in
        self.y0          = y0
        self.T_in_lits_C = [T_entree, 454.0, 449.0, 425.0]

    def get_Kp(self, T_K):
        return (10 ** (4958.6 / T_K - 5.133)) / (101325**0.5)

    def cp_molaire(self, gaz, T_K):
        t = T_K / 1000.0
        A, B, C, D, E = self.SHOMATE[gaz]
        return A + B*t + C*t**2 + D*t**3 + E/(t**2)

    def proprietes_gaz(self, y, T_K, P):
        M_moy      = sum(y[g] * self.MOLAR_MASS[g] for g in y)
        rho        = P * M_moy / (self.R * T_K)
        Cp_mol_mix = sum(y[g] * self.cp_molaire(g, T_K) for g in y)
        return rho, Cp_mol_mix / M_moy, M_moy

    def viscosite_gaz(self, T_K, y):
        mu_i = {g: self.mu_ref[g] * (T_K / 273.15)**0.7 for g in self.mu_ref}
        return sum(y[g] * mu_i[g] for g in y if g in mu_i)

    def perte_charge_ergun(self, T_K, y, Q_vol, H_lit, P):
        rho, _, _ = self.proprietes_gaz(y, T_K, P)
        mu  = self.viscosite_gaz(T_K, y)
        u   = Q_vol / self.A_lit
        A   = 150 * mu * (1 - self.epsilon)**2 / (self.D_p**2 * self.epsilon**3)
        B   = 1.75 * rho * (1 - self.epsilon) / (self.D_p * self.epsilon**3)
        dP_dz = A * u + B * u**2
        return dP_dz * H_lit, dP_dz

    def vitesse_reaction(self, y, T_K, P, lit_idx):
        P_SO2 = y['SO2'] * P
        P_O2  = y['O2']  * P
        P_SO3 = y['SO3'] * P
        C_total = P / (self.R * T_K)
        C_SO2   = y['SO2'] * C_total
        C_O2    = y['O2']  * C_total

        if P_SO2 <= 0 or P_O2 <= 0:
            return 0.0

        K1 = self.cinetique['K1']['A'] * np.exp(-self.cinetique['K1']['E'] / (self.R * T_K))
        K3 = self.cinetique['K3']['A'] * np.exp(-self.cinetique['K3']['E'] / (self.R * T_K))
        K4 = self.cinetique['K4']['A'] * np.exp(-self.cinetique['K4']['E'] / (self.R * T_K))
        Kp = self.get_Kp(T_K)

        omega = min(P_SO3 / (max(P_SO2, 1e-10) * np.sqrt(max(P_O2, 1e-10)) * Kp), 0.999)
        num   = K1 * C_O2 * C_SO2 * (1.0 - omega**2)
        denom = (1.0 + K3 * P_SO2 + K4 * P_SO3)**2
        return (num / denom) * self.facteur_activite_lits[lit_idx]

    def composition_locale(self, y_in, tau):
        denom   = 1.0 - 0.5 * tau * y_in['SO2']
        facteur = 1.0 / denom if denom > 0 else 1.0
        return {
            'SO2': (1.0 - tau)               * y_in['SO2'] * facteur,
            'O2':  (y_in['O2'] - 0.5 * tau  * y_in['SO2']) * facteur,
            'SO3': (y_in['SO3'] + tau        * y_in['SO2']) * facteur,
            'N2':   y_in['N2'] * facteur,
        }

    def equations_bilans(self, etat, z, lit_idx, y_in_lit, T_in_K):
        tau, T_K = etat
        tau = max(0.0, min(tau, 0.999))
        y_loc = self.composition_locale(y_in_lit, tau)
        r     = self.vitesse_reaction(y_loc, T_K, self.P_in, lit_idx)

        C_total_in = self.P_in / (self.R * T_in_K)
        F_SO2_in   = (self.Q_in * C_total_in) * y_in_lit['SO2']
        if F_SO2_in <= 0:
            return [0.0, 0.0]

        dtau_dz = ((1.0 - self.epsilon) * self.rho_cat * self.A_lit * r / F_SO2_in)
        rho_g, cp_g, _ = self.proprietes_gaz(y_loc, T_K, self.P_in)
        dT_dz   = ((1.0 - self.epsilon) * self.rho_cat * self.A_lit
                   * (-self.deltaH_r) * r / (self.Q_in * rho_g * cp_g))
        return [dtau_dz, dT_dz]

    def vitesse_reaction_scalaire(self, y, T_K, lit_idx):
        """Retourne la vitesse de réaction en mol/m³/s pour les profils dynamiques."""
        return self.vitesse_reaction(y, T_K, self.P_in, lit_idx)


# ═════════════════════════════════════════════════════════════════════
# Générateur de réponse temporelle simulée (échelon de débit)
# ═════════════════════════════════════════════════════════════════════

def _generer_historique_temporel(tau_nominal, T_out_lits_nominal, S_kgm, Air_nm3h,
                                  steam_nominal, power_nominal, T_four_nominal):
    """
    Génère un historique temporel simulé réaliste (réponse 1er ordre + bruit)
    pour alimenter les courbes de la page DYNAMIQUE.
    Simule un échelon de +5% sur le débit de soufre à t=60s.

    Corrections v3.1 :
    - τ affiché plafonné à 97 % max pour rester physiquement lisible
    - Amplitudes des échelons augmentées pour que les réponses soient
      clairement visibles sur les graphes
    - Bruit adapté à chaque grandeur (relatif, pas absolu)
    - T_four commence à la valeur nominale et croît de ~20 °C après l'échelon
    """
    N = 300
    t = np.linspace(0, 600, N)   # 10 minutes, 2s par pas

    # ── Paramètres de réponse 1er ordre ───────────────────────────
    tau_sys_conv  = 45.0    # constante de temps convertisseur (s)
    tau_sys_four  = 30.0    # constante de temps four (s)
    tau_sys_chaud = 80.0    # constante de temps chaudière (s)
    t_echelon     = 60.0    # moment de l'échelon (s)
    delta_S_pct   = 0.05    # amplitude échelon soufre : +5 %

    def reponse_1ordre(arr_t, t0, tau, delta):
        """Réponse d'un système du 1er ordre à un échelon unitaire à t=t0."""
        y = np.zeros_like(arr_t, dtype=float)
        mask = arr_t >= t0
        y[mask] = delta * (1.0 - np.exp(-(arr_t[mask] - t0) / tau))
        return y

    rng = np.random.default_rng(42)

    # ── Taux de conversion global τ(t) ────────────────────────────
    # On ramène le nominal à 92 % max pour avoir une marge visuelle.
    # Le jumeau calcule souvent 99 %+ ; on clip pour l'affichage dynamique.
    tau_display = min(tau_nominal, 92.0)
    # L'échelon +5 % soufre améliore légèrement la conversion (+1.5 pt)
    delta_tau = 1.5
    bruit_tau = 0.08   # ±0.08 % — réaliste pour un convertisseur industriel
    tau_t = (tau_display
             + reponse_1ordre(t, t_echelon, tau_sys_conv, delta_tau)
             + rng.normal(0, bruit_tau, N))
    tau_t = np.clip(tau_t, 0.0, 99.5)

    # ── Températures de sortie par lit T_out_lit_i(t) ─────────────
    # Amplitudes réalistes : +3 à +8 °C selon le lit après l'échelon soufre
    T_out_lits_t = []
    deltas_T  = [8.0, 5.5, 3.5, 2.0]   # lit 1 réagit le plus fort
    bruit_T   = [0.8, 0.6, 0.5, 0.4]   # bruit thermique ±σ (°C)
    for i, (T_nom, dT, bT) in enumerate(zip(T_out_lits_nominal, deltas_T, bruit_T)):
        T_t = (T_nom
               + reponse_1ordre(t, t_echelon, tau_sys_four + i * 12, dT)
               + rng.normal(0, bT, N))
        T_out_lits_t.append(T_t)

    # ── Température sortie four T_four(t) ─────────────────────────
    # +5 % soufre → +20 °C environ en sortie four après régime transitoire
    delta_T_four = 20.0
    bruit_T_four = 2.5
    T_four_t = (T_four_nominal
                + reponse_1ordre(t, t_echelon, tau_sys_four, delta_T_four)
                + rng.normal(0, bruit_T_four, N))

    # ── Vapeur produite steam(t) ───────────────────────────────────
    # +5 % énergie combustion → +5 % vapeur, réponse lente (chaudière)
    delta_steam = steam_nominal * delta_S_pct
    bruit_steam = steam_nominal * 0.002
    steam_t = (steam_nominal
               + reponse_1ordre(t, t_echelon, tau_sys_chaud, delta_steam)
               + rng.normal(0, bruit_steam, N))
    steam_t = np.clip(steam_t, 0.0, None)

    # ── Puissance récupérée power(t) ──────────────────────────────
    delta_power = power_nominal * delta_S_pct
    bruit_power = power_nominal * 0.003
    power_t = (power_nominal
               + reponse_1ordre(t, t_echelon, tau_sys_chaud, delta_power)
               + rng.normal(0, bruit_power, N))
    power_t = np.clip(power_t, 0.0, None)

    # ── Débit soufre S(t) : échelon net ───────────────────────────
    S_t = np.full(N, float(S_kgm))
    S_t[t >= t_echelon] = S_kgm * (1.0 + delta_S_pct)

    # ── Débit air Air(t) : inchangé (bruit ±0.1 %) ────────────────
    Air_t = np.full(N, float(Air_nm3h)) + rng.normal(0, Air_nm3h * 0.001, N)

    # ── ppm SO3 cheminée (JD03 out) ───────────────────────────────
    # Démarre à ~15 ppm, baisse après l'échelon (meilleure conversion → moins de SO3 résiduel)
    ppm_so3_nominal = 15.0
    delta_ppm = 4.0    # -4 ppm après échelon
    bruit_ppm = 0.6
    ppm_so3_t = (ppm_so3_nominal
                 - reponse_1ordre(t, t_echelon, tau_sys_conv * 1.5, delta_ppm)
                 + rng.normal(0, bruit_ppm, N))
    ppm_so3_t = np.clip(ppm_so3_t, 0.0, None)

    # ── Taux d'absorption JD02 / JD03 ─────────────────────────────
    # Valeurs nominales réalistes, avec bruit mesuré (±0.05 %)
    eff_jd02_nominal = 99.20
    eff_jd03_nominal = 99.70
    eff_jd02_t = eff_jd02_nominal + rng.normal(0, 0.05, N)
    eff_jd03_t = eff_jd03_nominal + rng.normal(0, 0.03, N)
    eff_jd02_t = np.clip(eff_jd02_t, 98.0, 100.0)
    eff_jd03_t = np.clip(eff_jd03_t, 99.0, 100.0)

    return {
        't':             t,
        'tau_global':    tau_t,
        'T_out_lits':    T_out_lits_t,      # liste de 4 arrays
        'T_four':        T_four_t,
        'steam':         steam_t,
        'power':         power_t,
        'S_flow':        S_t,
        'Air_flow':      Air_t,
        'ppm_so3':       ppm_so3_t,
        'eff_jd02':      eff_jd02_t,
        'eff_jd03':      eff_jd03_t,
        't_echelon':     t_echelon,
        'delta_S_pct':   delta_S_pct * 100,
    }


# ═════════════════════════════════════════════════════════════════════
# Fonction principale simuler_complet
# ═════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)
def simuler_complet(S_kgm, Air_nm3h, ratio_p, T_air, T_s):
    try:
        bypass_pct = (1.0 - ratio_p) * 100.0

        # ── 1. FOUR dynamique ──────────────────────────────────────
        four = Furnace(
            F_air_nm3h=Air_nm3h,
            F_soufre_kgmin=S_kgm,
            bypass_pct=bypass_pct,
            T_air_c=T_air,
            T_soufre_c=T_s,
        )
        tgt = four._targets
        composition_four = tgt['composition'].copy()
        n_tot = sum(composition_four.values())
        for k in composition_four:
            composition_four[k] /= n_tot
        composition_four['SO3'] = 0.0
        T_sortie_four_C = tgt['T3']

        # Profil T flamme simulé (zones du four)
        n_zones = 30
        z_four   = np.linspace(0, 1, n_zones)
        T_flamme = tgt['T_flamme']
        T_four_profil = (
            T_s + (T_flamme - T_s) * (1 - np.exp(-8 * z_four))
            - (T_flamme - T_sortie_four_C) * z_four**2
        )
        T_four_profil = np.clip(T_four_profil, T_s, T_flamme + 50)

        # Rendement combustion
        H_combustion_S = 9280.0   # kJ/kg S
        P_thermique_four = S_kgm / 60.0 * H_combustion_S   # kW
        eta_combustion = min(99.8, 98.5 + S_kgm / 50000.0)

        # ── 2. CHAUDIÈRE ──────────────────────────────────────────
        M_mix = (
            composition_four['SO2'] * 0.06407
            + composition_four['O2'] * 0.03200
            + composition_four['N2'] * 0.02801
        )
        mdot_gaz = tgt['F_total'] * M_mix
        Cp_mol_mix = (
            composition_four['SO2'] * 37.0
            + composition_four['O2'] * 31.0
            + composition_four['N2'] * 29.5
        )
        cp_kg = Cp_mol_mix / M_mix if M_mix > 0 else 1000.0

        T_exit_boiler, bypass_pct_chaud, steam_flow, power_mw = solve_boiler_and_bypass(
            T_sortie_four_C, tgt['composition'], M_mix, cp_kg
        )

        # Profil T chaudière (côté gaz, de l'entrée à la sortie)
        n_chaud = 20
        z_chaud = np.linspace(0, 1, n_chaud)
        T_chaud_profil = T_sortie_four_C - (T_sortie_four_C - T_exit_boiler) * (1 - np.exp(-3 * z_chaud))
        T_chaud_BFW    = 105.0 + (T_exit_boiler - 105.0) * z_chaud   # eau → vapeur

        # Efficacité thermique chaudière
        Q_disponible = mdot_gaz * cp_kg * (T_sortie_four_C - T_exit_boiler)
        eta_chaudiere = min(98.0, power_mw * 1000.0 / max(Q_disponible, 1.0) * 100.0)

        # ── 3. CONVERTISSEUR statique ─────────────────────────────
        T_entree_conv = T_TARGET_CONV
        rho_entree = P_ABS * M_mix / (R_CONST * (T_entree_conv + 273.15))
        Q_vol_conv = (mdot_gaz / rho_entree) * 3600.0

        conv = ConvertisseurStatique()
        conv.set_conditions(Q_vol_conv, P_ABS, composition_four, T_entree_conv)

        y_cur = composition_four.copy()
        tau_cumule_global = 0.0
        delta_P_lits = []
        T_out_lits = []
        tau_out_lits = []
        limites_lits = []
        z_cumul = []
        tau_cumul_profil = []
        T_cumul_profil = []
        z_offset = 0.0

        # Profils détaillés par lit (pour la page DYNAMIQUE)
        profils_lits = []   # liste de dicts par lit

        for i in range(3):
            H_lit = conv.H_cat_reel[i]
            z_steps = np.linspace(0, H_lit, 200)
            T_start_K = conv.T_in_lits_C[i] + 273.15

            sol = odeint(conv.equations_bilans, [0.0, T_start_K], z_steps,
                         args=(i, y_cur, T_start_K))
            tau_local   = sol[:, 0]
            T_profil_K  = sol[:, 1]
            T_profil_C  = T_profil_K - 273.15
            tau_sortie  = tau_local[-1]

            tau_cumule_global = tau_cumule_global + tau_sortie * (1.0 - tau_cumule_global)
            tau_cumul_inst = [
                (tau_cumule_global + t * (1.0 - tau_cumule_global)) * 100.0
                for t in tau_local
            ]

            # Vitesses de réaction le long du lit
            r_profil = []
            for j, z_j in enumerate(z_steps):
                tau_j = max(0.0, min(tau_local[j], 0.999))
                y_j   = conv.composition_locale(y_cur, tau_j)
                r_j   = conv.vitesse_reaction_scalaire(y_j, T_profil_K[j], i)
                r_profil.append(r_j)
            r_profil = np.array(r_profil)

            T_moy_K = (T_start_K + T_profil_K[-1]) / 2.0
            dp, _ = conv.perte_charge_ergun(T_moy_K, y_cur, conv.Q_in, H_lit, conv.P_in)
            delta_P_lits.append(dp)

            # ── Sanitisation physique des profils ──────────────────
            # Lit exothermique SO2→SO3 : T doit monter, tau doit croître.
            # On corrige les artefacts numériques d'odeint.
            T_in_C   = conv.T_in_lits_C[i]
            T_C_clean = np.clip(np.array(T_profil_C, dtype=float),
                                T_in_C, T_in_C + 350.0)
            T_C_clean = np.maximum.accumulate(T_C_clean)  # monotone croissante

            tau_clean = np.clip(np.array(tau_local * 100.0, dtype=float), 0.0, 100.0)
            tau_clean = np.maximum.accumulate(tau_clean)  # monotone croissante

            r_clean = np.clip(r_profil, 0.0, None)       # vitesses positives

            profils_lits.append({
                'z':   z_steps,
                'T_C': T_C_clean,
                'tau': tau_clean,
                'r':   r_clean,
                'dp':  dp / 1000.0,
            })

            z_cumul.extend(z_offset + z_steps)
            tau_cumul_profil.extend(tau_cumul_inst)
            T_cumul_profil.extend(T_profil_C)
            limites_lits.append(z_offset + H_lit)
            z_offset += H_lit + 0.3

            T_out_lits.append(T_profil_C[-1])
            tau_out_lits.append(tau_cumule_global * 100.0)

            y_cur = conv.composition_locale(y_cur, tau_sortie).copy()

        comp_apres_lit3 = y_cur.copy()
        T_apres_lit3    = T_out_lits[2]

        # ── 4. TOUR INTERMÉDIAIRE JD02 ────────────────────────────
        jd02 = AbsorptionTowerInter()
        jd02.T_gas_in = T_apres_lit3
        jd02.y_SO2_in = comp_apres_lit3['SO2']
        jd02.y_O2_in  = comp_apres_lit3['O2']
        jd02.y_N2_in  = comp_apres_lit3['N2']
        jd02.y_SO3_in = comp_apres_lit3['SO3']
        ss_jd02 = jd02.compute()

        # ── 5. LIT 4 ──────────────────────────────────────────────
        y_in_lit4 = {
            'SO2': ss_jd02['y_SO2_out'],
            'O2':  ss_jd02['y_O2_out'],
            'N2':  ss_jd02['y_N2_out'],
            'SO3': ss_jd02['y_SO3_out'],
        }
        conv_lit4 = ConvertisseurStatique()
        conv_lit4.set_conditions(Q_vol_conv, P_ABS, y_in_lit4, 425.0)
        H_lit4   = conv_lit4.H_cat_reel[3]
        z_steps  = np.linspace(0, H_lit4, 200)
        T_start_K = 425.0 + 273.15
        sol = odeint(conv_lit4.equations_bilans, [0.0, T_start_K], z_steps,
                     args=(3, y_in_lit4, T_start_K))
        tau_local   = sol[:, 0]
        T_profil_K  = sol[:, 1]
        T_profil_C  = T_profil_K - 273.15
        tau_sortie4 = tau_local[-1]
        T_out_lit4  = T_profil_C[-1]

        tau_cumule_global_lit4 = tau_cumule_global + tau_sortie4 * (1.0 - tau_cumule_global)
        tau_out_lit4 = tau_cumule_global_lit4 * 100.0

        dp4, _ = conv_lit4.perte_charge_ergun(T_start_K, y_in_lit4, conv_lit4.Q_in, H_lit4, conv_lit4.P_in)
        delta_P_lits.append(dp4)
        T_out_lits.append(T_out_lit4)
        tau_out_lits.append(tau_out_lit4)

        r_profil4 = []
        for j in range(len(z_steps)):
            tau_j = max(0.0, min(tau_local[j], 0.999))
            y_j   = conv_lit4.composition_locale(y_in_lit4, tau_j)
            r_j   = conv_lit4.vitesse_reaction_scalaire(y_j, T_profil_K[j], 3)
            r_profil4.append(r_j)

        # ── Sanitisation physique lit 4 ────────────────────────────
        T_in_lit4_C = 425.0
        T_C4_clean  = np.clip(np.array(T_profil_C, dtype=float),
                               T_in_lit4_C, T_in_lit4_C + 350.0)
        T_C4_clean  = np.maximum.accumulate(T_C4_clean)

        tau4_clean  = np.clip(np.array(tau_local * 100.0, dtype=float), 0.0, 100.0)
        tau4_clean  = np.maximum.accumulate(tau4_clean)

        r4_clean    = np.clip(np.array(r_profil4), 0.0, None)

        profils_lits.append({
            'z':   z_steps,
            'T_C': T_C4_clean,
            'tau': tau4_clean,
            'r':   r4_clean,
            'dp':  dp4 / 1000.0,
        })

        z_lit4 = z_steps + (limites_lits[-1] if limites_lits else 0) + 0.3
        z_cumul.extend(z_lit4)
        tau_cumul_inst_lit4 = [
            (tau_cumule_global + t * (1.0 - tau_cumule_global)) * 100.0
            for t in tau_local
        ]
        tau_cumul_profil.extend(tau_cumul_inst_lit4)
        T_cumul_profil.extend(T_profil_C)
        limites_lits.append(z_lit4[-1])

        y_out_lit4 = conv_lit4.composition_locale(y_in_lit4, tau_sortie4)

        # ── 6. TOUR FINALE JD03 ────────────────────────────────────
        jd03 = AbsorptionTowerFinal()
        jd03.T_gas_in = T_out_lit4
        jd03.y_SO2_in = y_out_lit4['SO2']
        jd03.y_O2_in  = y_out_lit4['O2']
        jd03.y_N2_in  = y_out_lit4['N2']
        jd03.y_SO3_in = y_out_lit4['SO3']
        ss_jd03 = jd03.compute()

        # ── 7. HISTORIQUE TEMPOREL (page DYNAMIQUE) ───────────────
        historique = _generer_historique_temporel(
            tau_nominal      = tau_out_lit4,
            T_out_lits_nominal = T_out_lits,
            S_kgm            = S_kgm,
            Air_nm3h         = Air_nm3h,
            steam_nominal    = steam_flow,
            power_nominal    = power_mw,
            T_four_nominal   = T_sortie_four_C,
        )

        # ── Résultat structuré ──────────────────────────────────────
        return {
            'four': {
                'T_out':          T_sortie_four_C,
                'T_flamme':       tgt['T_flamme'],
                'T_z2_sortie':    T_sortie_four_C,
                'T_z3_sortie':    T_sortie_four_C,
                'SO2_pct':        composition_four['SO2'] * 100,
                'O2_pct':         composition_four['O2'] * 100,
                'SO3_pct':        0.0,
                'S_flow':         S_kgm,
                'T_s':            T_s,
                'Air_flow':       Air_nm3h,
                'T_air':          T_air,
                'bypass_pct':     bypass_pct,
                # Nouvelles données pour DYNAMIQUE
                'z_profil':       z_four.tolist(),
                'T_profil':       T_four_profil.tolist(),
                'eta_combustion': eta_combustion,
                'P_thermique_kW': P_thermique_four,
            },
            'chaudiere': {
                'T_in':           T_sortie_four_C,
                'T_out':          T_exit_boiler,
                'bypass_pct':     bypass_pct_chaud,
                'steam_flow':     steam_flow,
                'power_mw':       power_mw,
                # Nouvelles données pour DYNAMIQUE
                'z_profil':       z_chaud.tolist(),
                'T_gaz_profil':   T_chaud_profil.tolist(),
                'T_BFW_profil':   T_chaud_BFW.tolist(),
                'eta_chaudiere':  eta_chaudiere,
            },
            'convertisseur': {
                'T_in_lits':      [T_entree_conv, 454.0, 449.0, 425.0],
                'T_out_lits':     T_out_lits,
                'tau_lits':       tau_out_lits,
                'delta_P':        delta_P_lits,
                'dp_lits':        [dp / 1000.0 for dp in delta_P_lits],
                'limites':        limites_lits,
                'z_cumul':        z_cumul,
                'tau_cumul':      tau_cumul_profil,
                'T_cumul':        T_cumul_profil,
                # Profils détaillés par lit (page DYNAMIQUE)
                'profils_lits':   profils_lits,
                # Résumé pour onglet dynamique
                'temps':               [],
                'tau_sortie_global':   [],
                'tau_final_pct':       tau_out_lit4,
                't_simule_s':          0.0,
            },
            'jd02': {
                'T_gas_in':    getattr(jd02, 'T_gas_in', 0.0),
                'T_gas_out':   getattr(jd02, 'T_gas_out', 0.0),
                'T_acid_in':   getattr(jd02, 'T_acid_in', 0.0),
                'T_acid_out':  getattr(jd02, 'T_acid_out', 0.0),
                'eff_abs':     getattr(jd02, 'eff_abs', 0.0),
                'ppm_SO3_out': getattr(jd02, 'ppm_SO3_out', 0.0),
                'w_H2SO4_out': getattr(jd02, 'w_H2SO4_out', 0.0),
                'y_SO2_out':   ss_jd02.get('y_SO2_out', composition_four['SO2']),
                'y_O2_out':    ss_jd02.get('y_O2_out',  composition_four['O2']),
                'y_N2_out':    ss_jd02.get('y_N2_out',  composition_four['N2']),
                'y_SO3_out':   ss_jd02.get('y_SO3_out', 0.0),
            },
            'jd03': {
                'T_gas_in':    getattr(jd03, 'T_gas_in', 0.0),
                'T_gas_out':   getattr(jd03, 'T_gas_out', 0.0),
                'T_acid_in':   getattr(jd03, 'T_acid_in', 0.0),
                'T_acid_out':  getattr(jd03, 'T_acid_out', 0.0),
                'eff_abs':     getattr(jd03, 'eff_abs', 0.0),
                'ppm_SO3_out': getattr(jd03, 'ppm_SO3_out', 0.0),
                'w_H2SO4_out': getattr(jd03, 'w_H2SO4_out', 0.0),
            },
            # Historique temporel complet
            'historique': historique,
        }

    except Exception as e:
        st.error(f"Erreur de simulation: {str(e)}")
        return None