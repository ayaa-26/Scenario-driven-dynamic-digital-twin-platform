#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tour de Séchage — Jumeau Numérique OCP
Procédé de contact à double absorption — Unité d'acide sulfurique
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ── Constantes inline (remplace core.constants) ──────────────────────
R_CONST        = 8.314          # J/(mol·K)
g_GRAV         = 9.81           # m/s²
P_ATM          = 101_325.0      # Pa
M_H2O          = 0.018015       # kg/mol
M_H2SO4        = 0.098079       # kg/mol
M_O2           = 0.031999       # kg/mol
M_N2           = 0.028014       # kg/mol
DELTA_H_ABS_H2O   = 2_501_000.0   # J/kg
RHO_H2SO4_986     = 1_836.0        # kg/m³
MU_H2SO4_986      = 0.02           # Pa·s
SIGMA_H2SO4       = 0.0735         # N/m
CP_H2SO4_986      = 1_470.0        # J/(kg·K)
LAMBDA_H2SO4      = 0.35           # W/(m·K)
DL_H2O_H2SO4      = 1.5e-9        # m²/s


class DryingTower:
    """
    Tour de séchage à garnissage Intalox céramique 2 pouces.
    Version autonome — sans dépendances models.base / core.constants
    """
    # ── Géométrie colonne
    D_COL    = 7.6      # m
    D_P      = 0.0508   # m
    A_P      = 118.0    # m²/m³
    EPSILON  = 0.79
    SIGMA_C  = 0.061    # N/m
    H_PACK   = 3.7      # m
    # ── Propriétés gaz
    MU_G     = 1.87e-5  # Pa·s
    CP_G     = 1_012.0  # J/(kg·K)
    LAMBDA_G = 0.0263   # W/(m·K)
    DG_H2O   = 2.82e-5  # m²/s
    A_ONDA   = 5.23

    def __init__(self, name="Tour de séchage", tag="401AD02"):
        self.name = name
        self.tag  = tag
        # ── Paramètres opératoires nominaux OCP
        self.Q_gas_Nm3h  = 368_665.0
        self.Q_acid_m3h  = 1_245.0
        self.T_gas_in    = 30.0
        self.T_acid_in   = 66.0
        self.w_H2SO4_in  = 0.986
        # ── Composition gaz entrant
        self.y_O2_in  = 76_235.0  / 368_665.0
        self.y_N2_in  = 287_699.0 / 368_665.0
        self.y_H2O_in = 4_731.0   / 368_665.0
        # ── Sorties
        self.T_gas_out  = self.T_gas_in
        self.T_acid_out = self.T_acid_in
        self.y_H2O_out  = self.y_H2O_in
        self.w_out_gNm3 = 0.0
        self.eff_abs    = 0.0
        self.pressure_drop_active = False
        # ── Résultats
        self.ss_results   = {}
        self.htu_results  = {}
        self.coeffs       = {}
        self.dyn_results  = {}
        self.param_results= {}
        # ── Historiques
        self.time_hist      = []
        self.T_gas_out_hist = []
        self.y_H2O_out_hist = []
        self.eff_hist       = []
        self.history        = []

    # ─────────────────────────────────────────────────────────────────
    # Physique privée
    # ─────────────────────────────────────────────────────────────────
    def _compute_geometry(self):
        M_G_mix = (self.y_O2_in * M_O2 + self.y_N2_in * M_N2
                   + self.y_H2O_in * M_H2O)
        T_G_K  = self.T_gas_in + 273.15
        P_op   = P_ATM
        rho_G  = P_op * M_G_mix / (R_CONST * T_G_K)
        A_col  = np.pi * (self.D_COL / 2.0) ** 2
        F_G_mol= self.Q_gas_Nm3h / 3600.0 / 22.414e-3
        M_mix_L= self.w_H2SO4_in * M_H2SO4 + (1 - self.w_H2SO4_in) * M_H2O
        m_dot_L= self.Q_acid_m3h * RHO_H2SO4_986 / 3600.0
        F_L_mol= m_dot_L / M_mix_L
        m_dot_G= F_G_mol * M_G_mix
        return {
            'P_op': P_op, 'T_G_K': T_G_K,
            'M_G_mix': M_G_mix, 'rho_G': rho_G,
            'A_col': A_col, 'D_col': self.D_COL,
            'F_G_mol': F_G_mol,
            'M_mix_L': M_mix_L, 'm_dot_L': m_dot_L, 'F_L_mol': F_L_mol,
            'G_mol': F_G_mol / A_col,
            'L_mol': F_L_mol / A_col,
            'G_mass': m_dot_G / A_col,
            'L_mass': m_dot_L / A_col,
            'c_G_tot': P_op / (R_CONST * T_G_K),
            'c_L_tot': RHO_H2SO4_986 / M_mix_L,
            'x_H2O_in': 1.0 - self.w_H2SO4_in * M_H2SO4 / M_mix_L,
        }

    def _equilibrium(self, geo):
        CORR = 0.85
        Td = np.array([343.15, 353.15, 363.15])
        pd = np.array([0.54, 1.15, 2.34]) * CORR
        X  = np.column_stack([np.ones(3), 1.0 / Td])
        c  = np.linalg.lstsq(X, np.log(pd), rcond=None)[0]
        A_ant, B_ant = c[0], -c[1]
        def p_eq(T_C):
            return np.exp(A_ant - B_ant / (np.asarray(T_C, float) + 273.15))
        def y_star(T_C):
            return np.minimum(p_eq(T_C) / geo['P_op'], 1.0)
        def m_slope(T_C):
            return p_eq(T_C) / (geo['P_op'] * max(geo['x_H2O_in'], 1e-4))
        return p_eq, y_star, m_slope

    def _onda(self, geo, y_star_fn, m_slope_fn):
        G_mass = geo['G_mass']; L_mass = geo['L_mass']
        Sc_G = self.MU_G / (geo['rho_G'] * self.DG_H2O)
        Sc_L = MU_H2SO4_986 / (RHO_H2SO4_986 * DL_H2O_H2SO4)
        Pr_G = self.MU_G * self.CP_G / self.LAMBDA_G
        Pr_L = MU_H2SO4_986 * CP_H2SO4_986 / LAMBDA_H2SO4
        Re_aw = L_mass / (self.A_P * MU_H2SO4_986)
        Fr_aw = L_mass**2 * self.A_P / (RHO_H2SO4_986**2 * g_GRAV)
        We_aw = L_mass**2 / (RHO_H2SO4_986 * SIGMA_H2SO4 * self.A_P)
        arg = (-1.45 * (self.SIGMA_C / SIGMA_H2SO4)**0.75
               * Re_aw**0.1 * Fr_aw**(-0.05) * We_aw**0.2)
        a_w = np.clip(1.0 - np.exp(arg), 0.3, 1.0) * self.A_P
        Re_G   = G_mass / (self.A_P * self.MU_G)
        kG_p   = (self.A_ONDA * Re_G**0.7 * Sc_G**(1/3)
                  * (self.A_P * self.D_P)**(-2.0)
                  * (self.A_P * self.DG_H2O) / (R_CONST * geo['T_G_K']))
        kG_mol = kG_p * geo['P_op']
        Re_L_kL= L_mass / (a_w * MU_H2SO4_986)
        kL_p   = (0.0051 * Re_L_kL**(2/3) * Sc_L**(-0.5)
                  * (self.A_P * self.D_P)**0.4
                  * (g_GRAV * MU_H2SO4_986 / RHO_H2SO4_986)**(1/3))
        kL_mol = kL_p * geo['c_L_tot']
        m_op   = float(m_slope_fn(self.T_acid_in))
        R_gaz  = 1.0 / kG_mol
        R_liq  = m_op / kL_mol
        KG     = 1.0 / (R_gaz + R_liq)
        KL     = KG / m_op if m_op > 1e-12 else kL_mol
        pct_g  = R_gaz / (R_gaz + R_liq) * 100.0
        Le_G   = Sc_G / Pr_G; Le_L = Sc_L / Pr_L
        h_G    = (kG_p * R_CONST * geo['T_G_K']
                  * geo['rho_G'] * self.CP_G * Le_G**(-2/3) / geo['M_G_mix'])
        h_L    = kL_p * RHO_H2SO4_986 * CP_H2SO4_986 * Le_L**(-2/3)
        U_eff  = 1.0 / (1.0 / h_G + 1.0 / h_L)
        return {
            'a_w': a_w, 'kG_mol': kG_mol, 'kL_mol': kL_mol,
            'KG_tot': KG, 'KL_tot': KL, 'm_op': m_op,
            'pct_gaz': pct_g, 'U_eff': U_eff,
            'Re_G': Re_G, 'Sc_G': Sc_G,
            'R_gaz': R_gaz, 'R_liq': R_liq,
        }

    def _htu_ntu(self, geo, coef, y_star_fn):
        y_target = 0.01e-3 * 22.414e-3 / M_H2O
        y_s   = float(y_star_fn(68.0))
        y_arr = np.linspace(y_target, self.y_H2O_in, 500)
        denom = np.maximum(y_arr - y_s, 1e-15)
        integ = (1.0 - y_s) / ((1.0 - y_arr) * denom)
        integ = np.where(y_arr > y_s + 1e-12, integ, 0.0)
        NOG   = float(np.trapezoid(integ, y_arr))
        KG, aw= coef['KG_tot'], coef['a_w']
        KL, m = coef['KL_tot'], coef['m_op']
        Gm, Lm= geo['G_mol'], geo['L_mol']
        HG    = Gm / (KG * aw)
        HL    = Lm / (KL * aw) if KL > 1e-6 else 1e4
        lam   = m * Gm / Lm
        HOG   = HG + lam * HL
        HT    = HOG * NOG
        eta   = self.H_PACK / HT if HT > 0 else 0.0
        return {
            'NOG': NOG, 'HOG': HOG, 'HG': HG, 'HL': HL,
            'HT_req': HT, 'eta_pack': eta,
            'y_target': y_target, 'y_star': y_s, 'lambda': lam,
        }

    def _bvp(self, geo, coef, y_star_fn):
        KG, aw, Ue = coef['KG_tot'], coef['a_w'], coef['U_eff']
        Gm, Lm     = geo['G_mol'], geo['L_mol']
        Gkg, Lkg   = geo['G_mass'], geo['L_mass']
        def odes(z, Y):
            y_v, x_v, TG, TL = Y
            TL_c = float(np.clip(float(TL), 50.0, 110.0))
            y_s  = float(y_star_fn(TL_c))
            NA   = KG * aw * max(float(y_v) - y_s, 0.0)
            return [
                -NA / Gm,
                -NA / Lm,
                -Ue * aw / (Gkg * self.CP_G) * (float(TG) - float(TL)),
                (Ue * aw / (Lkg * CP_H2SO4_986) * (float(TG) - float(TL))
                 + NA * DELTA_H_ABS_H2O / (Lkg * CP_H2SO4_986)),
            ]
        def residuals(guess):
            TL_b, x_b = guess
            Y0  = [self.y_H2O_in, max(x_b, 1e-8), self.T_gas_in, TL_b]
            sol = solve_ivp(odes, [0, self.H_PACK], Y0, method='RK45',
                            max_step=self.H_PACK / 500, rtol=1e-6, atol=1e-9)
            Yh  = sol.y[:, -1]
            return [Yh[3] - self.T_acid_in, Yh[1] - geo['x_H2O_in']]
        try:
            result    = fsolve(residuals,
                               [self.T_acid_in + 5.0, geo['x_H2O_in'] * 0.5],
                               full_output=True)
            root      = np.asarray(result[0], dtype=float)
            TL_b      = float(root[0])
            x_b       = float(root[1])
            fvec      = np.asarray(result[1].get('fvec') or [1.0, 1.0], dtype=float)
            converged = bool(float(np.max(np.abs(fvec))) < 1e-3)
        except Exception:
            TL_b, x_b, converged = self.T_acid_in + 3.0, geo['x_H2O_in'] * 0.5, False
        z_arr  = np.linspace(0, self.H_PACK, 400)
        Y0     = [self.y_H2O_in, max(x_b, 1e-8), self.T_gas_in, TL_b]
        sol_f  = solve_ivp(odes, [0, self.H_PACK], Y0, t_eval=z_arr,
                           method='RK45', max_step=self.H_PACK / 500,
                           rtol=1e-7, atol=1e-11)
        y_ss   = np.clip(sol_f.y[0], 0, 1)
        TG_ss  = sol_f.y[2]
        TL_ss  = sol_f.y[3]
        y_out  = float(y_ss[-1])
        return {
            'z': z_arr, 'y': y_ss, 'TG': TG_ss, 'TL': TL_ss,
            'y_out': y_out,
            'TG_out': float(TG_ss[-1]),
            'TL_out': float(TL_ss[0]),
            'w_gNm3': y_out * M_H2O * 1000.0 / 22.414e-3,
            'eff': (self.y_H2O_in - y_out) / max(self.y_H2O_in, 1e-15) * 100.0,
            'converged': converged,
        }

    def _pressure_drop(self, geo, z_arr, TG_arr):
        eps = self.EPSILON; dp = self.D_P
        Q_gas_m3s = self.Q_gas_Nm3h / 3600.0 * (geo['T_G_K'] / 273.15)
        U_S_G     = Q_gas_m3s / geo['A_col']
        M_G_mix   = geo['M_G_mix']
        dpdz_arr  = np.zeros(len(z_arr))
        for i, TG in enumerate(TG_arr):
            rho_G_loc   = P_ATM * M_G_mix / (R_CONST * (float(TG) + 273.15))
            dpdz_arr[i] = (
                150.0 * self.MU_G * (1.0 - eps)**2 / (eps**3 * dp**2) * U_S_G
                + 1.75 * rho_G_loc * (1.0 - eps) / (eps**3 * dp) * U_S_G**2)
        delta_P = float(np.trapezoid(dpdz_arr, z_arr))
        dz      = float(z_arr[1] - z_arr[0]) if len(z_arr) > 1 else 1.0
        P_arr   = np.zeros(len(z_arr))
        P_arr[-1] = P_ATM
        for i in range(len(z_arr) - 2, -1, -1):
            P_arr[i] = P_arr[i + 1] + dpdz_arr[i] * dz
        return {
            'delta_P': delta_P,
            'delta_P_mbar': delta_P / 100.0,
            'delta_P_mmH2O': delta_P / 9.81,
            'P_profile': P_arr,
            'dpdz_profile': dpdz_arr,
        }

    # ─────────────────────────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────────────────────────
    def compute(self):
        geo = self._compute_geometry()
        _, y_star_fn, m_slope_fn = self._equilibrium(geo)
        coef = self._onda(geo, y_star_fn, m_slope_fn)
        ss   = self._bvp(geo, coef, y_star_fn)
        htu  = self._htu_ntu(geo, coef, y_star_fn)
        if self.pressure_drop_active:
            pdc = self._pressure_drop(geo, ss['z'], ss['TG'])
            ss.update(pdc)
        self.T_gas_out  = ss['TG_out']
        self.T_acid_out = ss['TL_out']
        self.y_H2O_out  = ss['y_out']
        self.w_out_gNm3 = ss['w_gNm3']
        self.eff_abs    = ss['eff']
        self.ss_results = ss
        self.htu_results= htu
        self.coeffs     = coef
        return ss

    def step(self, dt, time):
        self.compute()
        self.time_hist.append(time)
        self.T_gas_out_hist.append(self.T_gas_out)
        self.y_H2O_out_hist.append(self.y_H2O_out)
        self.eff_hist.append(self.eff_abs)
        if len(self.time_hist) > 300:
            self.time_hist.pop(0); self.T_gas_out_hist.pop(0)
            self.y_H2O_out_hist.pop(0); self.eff_hist.pop(0)
        self.state = {
            'time': time, 'T_gas_out': self.T_gas_out,
            'T_acid_out': self.T_acid_out, 'y_H2O_out': self.y_H2O_out,
            'w_gNm3': self.w_out_gNm3, 'eff_abs': self.eff_abs}
        return self.state

    def reset(self):
        self.T_gas_out = self.T_gas_in; self.T_acid_out = self.T_acid_in
        self.y_H2O_out = self.y_H2O_in; self.w_out_gNm3 = 0.0; self.eff_abs = 0.0
        self.time_hist = []; self.T_gas_out_hist = []
        self.y_H2O_out_hist = []; self.eff_hist = []
        self.ss_results = {}; self.htu_results = {}
        self.coeffs = {}; self.dyn_results = {}; self.param_results = {}
        self.history = []
