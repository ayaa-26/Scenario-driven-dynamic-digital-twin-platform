# =====================================================================
# model/absorption.py
# Tour d'absorption SO3 — Jumeau Numérique OCP
# Adapté pour intégration directe (sans models.base ni core.constants)
# Physique : Eq. 105–109 du rapport PFE (corrélation d'Onda 1968)
# JD02 : Tour intermédiaire (entre lit 3 et lit 4)
# JD03 : Tour finale      (après lit 4)
# =====================================================================
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ── Constantes (reprises de four.py pour cohérence) ──────────────────
R_CONST  = 8.314
P_ATM    = 101325.0
g_GRAV   = 9.81

# ── Masses molaires [kg/mol] ─────────────────────────────────────────
M_H2O   = 0.01802
M_H2SO4 = 0.09808
M_SO2   = 0.06407
M_SO3   = 0.08006
M_O2    = 0.03200
M_N2    = 0.02801

# ── Enthalpie d'absorption SO3 + H2O → H2SO4 ────────────────────────
DELTA_H_ABS_SO3 = 130_000.0          # J/mol SO3

# ── Propriétés H2SO4 98.5 % à 80 °C ─────────────────────────────────
RHO_H2SO4_985  = 1_800.0             # kg/m³
MU_H2SO4_985   = 4.5e-3              # Pa·s
SIGMA_H2SO4_985 = 0.053              # N/m
CP_H2SO4_985   = 1_520.0             # J/(kg·K)
LAMBDA_H2SO4_985 = 0.47              # W/(m·K)
DL_SO3_H2SO4   = 1.5e-9             # m²/s
DL_H2O_H2SO4   = 1.5e-9             # m²/s

# ── Constante de Henry SO3 / H2SO4 concentré ─────────────────────────
H_SO3_DEFAULT  = 0.5                 # Pa·m³/mol
EPS_L_HOLD     = 0.04               # rétention liquide [-]


# =====================================================================
class AbsorptionTower:
    """
    Tour d'absorption SO3 à garnissage Intalox céramique 3".
    Bilans de masse et d'énergie Eq. 105–109 (rapport PFE).
    Convention axiale : z=0 = bas (entrée gaz), z=H = haut (entrée acide).
    """

    # ── Géométrie garnissage (Intalox 3" / 76.2 mm) ──────────────────
    D_P      = 0.0762    # m
    A_P      = 69.0      # m²/m³
    EPSILON  = 0.78      # porosité totale
    SIGMA_C  = 0.061     # N/m (tension critique céramique)
    A_ONDA   = 5.23      # constante Onda (dp ≥ 25 mm)
    E_CHEM   = 1.50      # facteur accélération chimique

    # ── Propriétés gaz (~190 °C) ─────────────────────────────────────
    MU_G     = 2.2e-5    # Pa·s
    CP_G     = 900.0     # J/(kg·K)
    LAMBDA_G = 0.035     # W/(m·K)
    DG_SO3   = 2.1e-5    # m²/s

    # ── Sous-classes : géométrie colonne (redéfinie par JD02/JD03) ───
    D_COL  = 7.0
    H_PACK = 3.7

    def __init__(self, name="Tour intermédiaire", tag="JD02"):
        self.name = name
        self.tag  = tag

        # Paramètres opératoires par défaut (JD02)
        self.Q_gas_Nm3h  = 118_500.0
        self.Q_acid_m3h  = 1_065.0
        self.T_gas_in    = 165.0
        self.T_acid_in   = 80.0
        self.w_H2SO4_in  = 0.985
        self.H_SO3       = H_SO3_DEFAULT
        self.E_CHEM      = self.__class__.E_CHEM
        self.P_op        = P_ATM + 2_466.0 * g_GRAV   # ≈ 125 516 Pa

        self.y_SO3_in = 0.11172
        self.y_SO2_in = 0.00442
        self.y_O2_in  = 0.04917
        self.y_N2_in  = 1.0 - self.y_SO3_in - self.y_SO2_in - self.y_O2_in

        # Sorties (initialisées = entrées)
        self.T_gas_out    = self.T_gas_in
        self.T_acid_out   = self.T_acid_in
        self.y_SO3_out    = self.y_SO3_in
        self.C_SO3_out    = 0.0
        self.C_H2O_out    = 0.0
        self.w_H2SO4_out  = self.w_H2SO4_in
        self.eff_abs      = 0.0
        self.ppm_SO3_out  = self.y_SO3_in * 1e6
        self.ss_results   = {}

    # ------------------------------------------------------------------
    # Géométrie et débits
    # ------------------------------------------------------------------
    def _compute_geometry(self):
        M_G_mix = (self.y_SO3_in * M_SO3 + self.y_SO2_in * M_SO2
                   + self.y_O2_in  * M_O2  + self.y_N2_in  * M_N2)
        T_G_K  = self.T_gas_in + 273.15
        P_op   = self.P_op
        rho_G  = P_op * M_G_mix / (R_CONST * T_G_K)
        c_G_tot = P_op / (R_CONST * T_G_K)

        A_col  = np.pi * (self.D_COL / 2.0) ** 2
        Q_gas_m3s = self.Q_gas_Nm3h / 3600.0 * (T_G_K / 273.15)
        U_S_G  = Q_gas_m3s / A_col
        F_G_mol = self.Q_gas_Nm3h / 3600.0 / 22.414e-3
        C_SO3_in = self.y_SO3_in * c_G_tot

        w_H2O  = 1.0 - self.w_H2SO4_in
        x_H2O_in = ((w_H2O / M_H2O)
                    / (w_H2O / M_H2O + self.w_H2SO4_in / M_H2SO4))
        x_H2SO4_in = 1.0 - x_H2O_in
        M_mix_L = x_H2O_in * M_H2O + x_H2SO4_in * M_H2SO4
        c_L_tot = RHO_H2SO4_985 / M_mix_L
        C_H2O_in = x_H2O_in * c_L_tot

        m_dot_L  = self.Q_acid_m3h * RHO_H2SO4_985 / 3600.0
        F_L_mol  = m_dot_L / M_mix_L
        m_dot_G  = F_G_mol * M_G_mix
        Q_acid_m3s = self.Q_acid_m3h / 3600.0
        U_S_L    = Q_acid_m3s / A_col

        eps_L = min(EPS_L_HOLD, self.EPSILON * 0.08)
        eps_G = max(self.EPSILON - eps_L, 0.60)

        return {
            'P_op': P_op, 'T_G_K': T_G_K,
            'M_G_mix': M_G_mix, 'rho_G': rho_G,
            'A_col': A_col,
            'F_G_mol': F_G_mol, 'F_L_mol': F_L_mol,
            'm_dot_G': m_dot_G, 'm_dot_L': m_dot_L,
            'G_mol': F_G_mol / A_col,
            'L_mol': F_L_mol / A_col,
            'G_mass': m_dot_G / A_col,
            'L_mass': m_dot_L / A_col,
            'c_G_tot': c_G_tot, 'c_L_tot': c_L_tot,
            'C_SO3_in': C_SO3_in, 'C_H2O_in': C_H2O_in,
            'x_H2O_in': x_H2O_in, 'M_mix_L': M_mix_L,
            'U_S_G': U_S_G, 'U_S_L': U_S_L,
            'eps_G': eps_G, 'eps_L': eps_L,
        }

    # ------------------------------------------------------------------
    # Corrélation d'Onda (1968)
    # ------------------------------------------------------------------
    def _onda(self, geo):
        G_mass = geo['G_mass']
        L_mass = geo['L_mass']

        Sc_G = self.MU_G / (geo['rho_G'] * self.DG_SO3)
        Sc_L = MU_H2SO4_985 / (RHO_H2SO4_985 * DL_SO3_H2SO4)
        Pr_G = self.MU_G * self.CP_G / self.LAMBDA_G
        Pr_L = MU_H2SO4_985 * CP_H2SO4_985 / LAMBDA_H2SO4_985

        Re_aw = L_mass / (self.A_P * MU_H2SO4_985)
        Fr_aw = L_mass**2 * self.A_P / (RHO_H2SO4_985**2 * g_GRAV)
        We_aw = L_mass**2 / (RHO_H2SO4_985 * SIGMA_H2SO4_985 * self.A_P)
        arg   = (-1.45 * (self.SIGMA_C / SIGMA_H2SO4_985)**0.75
                 * Re_aw**0.1 * Fr_aw**(-0.05) * We_aw**0.2)
        a_w   = np.clip(1.0 - np.exp(arg), 0.3, 1.0) * self.A_P

        Re_G  = G_mass / (self.A_P * self.MU_G)
        kG_p  = (self.A_ONDA * Re_G**0.7 * Sc_G**(1/3)
                 * (self.A_P * self.D_P)**(-2.0)
                 * (self.A_P * self.DG_SO3) / (R_CONST * geo['T_G_K']))

        Re_L  = L_mass / (a_w * MU_H2SO4_985)
        kL_ms = (0.0051 * Re_L**(2/3) * Sc_L**(-0.5)
                 * (self.A_P * self.D_P)**0.4
                 * (g_GRAV * MU_H2SO4_985 / RHO_H2SO4_985)**(1/3))

        K_overall = 1.0 / (1.0 / kL_ms + 1.0 / (kG_p * self.H_SO3))

        Le_G  = Sc_G / Pr_G
        Le_L  = Sc_L / Pr_L
        h_G   = (kG_p * R_CONST * geo['T_G_K']
                 * geo['rho_G'] * self.CP_G * Le_G**(-2/3) / geo['M_G_mix'])
        h_L   = kL_ms * RHO_H2SO4_985 * CP_H2SO4_985 * Le_L**(-2/3)
        U_eff = 1.0 / (1.0 / h_G + 1.0 / h_L)

        return {
            'a_w': a_w, 'kG_p': kG_p, 'kL_ms': kL_ms,
            'K_overall': K_overall, 'U_eff': U_eff,
            'Re_G': Re_G, 'Sc_G': Sc_G, 'h_G': h_G, 'h_L': h_L,
        }

    # ------------------------------------------------------------------
    # Flux de transfert — Eq. 107
    # ------------------------------------------------------------------
    def _flux_N_SO3(self, C_SO3_G, _C_H2O_L, T_G_C, coef):
        T_G_K = float(T_G_C) + 273.15
        P_SO3 = max(float(C_SO3_G), 0.0) * R_CONST * T_G_K
        return self.E_CHEM * coef['kG_p'] * P_SO3

    # ------------------------------------------------------------------
    # BVP stationnaire — Eq. 105-109 (∂/∂t = 0)
    # ------------------------------------------------------------------
    def _bvp(self, geo, coef):
        a_w   = coef['a_w']
        U_eff = coef['U_eff']
        U_S_G = geo['U_S_G']
        U_S_L = geo['U_S_L']
        rho_G = geo['rho_G']

        def odes(_z, Y):
            CG, CH, TG, TL = Y
            CG = max(CG, 0.0);  CH = max(CH, 0.0)
            TG = float(np.clip(TG, 50.0, 300.0))
            TL = float(np.clip(TL, 50.0, 200.0))
            N_SO3  = self._flux_N_SO3(CG, CH, TG, coef)
            dCG_dz = -N_SO3 * a_w / U_S_G
            dCH_dz = +N_SO3 * a_w / U_S_L
            dTG_dz = -U_eff * a_w * (TG - TL) / (U_S_G * rho_G * self.CP_G)
            dTL_dz = -(U_eff * a_w * (TG - TL) + N_SO3 * a_w * DELTA_H_ABS_SO3) \
                     / (U_S_L * RHO_H2SO4_985 * CP_H2SO4_985)
            return [dCG_dz, dCH_dz, dTG_dz, dTL_dz]

        C_SO3_in = geo['C_SO3_in']
        C_H2O_in = geo['C_H2O_in']

        def residuals(guess):
            CH_bot = max(float(guess[0]), 0.0)
            TL_bot = float(guess[1])
            Y0  = [C_SO3_in, CH_bot, self.T_gas_in, TL_bot]
            sol = solve_ivp(odes, [0.0, self.H_PACK], Y0, method='Radau',
                            max_step=self.H_PACK / 100, rtol=1e-4, atol=1e-7)
            return [sol.y[1, -1] - C_H2O_in, sol.y[3, -1] - self.T_acid_in]

        F_SO3_in = geo['F_G_mol'] * self.y_SO3_in
        F_H2O_in = geo['F_L_mol'] * geo['x_H2O_in']
        frac_consumed  = min(F_SO3_in / max(F_H2O_in, 1.0), 0.9)
        CH_bot_guess   = C_H2O_in * (1.0 - frac_consumed)
        TL_bot_guess   = self.T_acid_in + 25.0 + frac_consumed * 40.0

        try:
            result    = fsolve(residuals, [CH_bot_guess, TL_bot_guess],
                               full_output=True, xtol=1e-4)
            CH_bot    = max(float(result[0][0]), 0.0)
            TL_bot    = float(result[0][1])
            fvec      = np.asarray(result[1].get('fvec', [99.0, 99.0]), dtype=float)
            converged = (int(result[2]) == 1) or bool(np.max(np.abs(fvec)) < 5.0)
        except Exception:
            CH_bot    = CH_bot_guess
            TL_bot    = TL_bot_guess
            converged = False

        z_arr = np.linspace(0.0, self.H_PACK, 400)
        Y0    = [C_SO3_in, CH_bot, self.T_gas_in, TL_bot]
        sol_f = solve_ivp(odes, [0.0, self.H_PACK], Y0, t_eval=z_arr,
                          method='Radau', max_step=self.H_PACK / 100,
                          rtol=1e-5, atol=1e-9)

        CG_ss  = np.clip(sol_f.y[0], 0.0, geo['c_G_tot'])
        CH_ss  = np.clip(sol_f.y[1], 0.0, C_H2O_in * 1.05)
        TG_ss  = sol_f.y[2]
        TL_ss  = sol_f.y[3]
        y_ss   = CG_ss / geo['c_G_tot']
        y_out  = float(y_ss[-1])
        eff    = (self.y_SO3_in - y_out) / max(self.y_SO3_in, 1e-15) * 100.0

        denom_G       = max(1.0 - self.y_SO3_in + y_out, 1e-12)
        y_SO3_out_true = y_out            / denom_G
        y_SO2_out_true = self.y_SO2_in   / denom_G
        y_O2_out_true  = self.y_O2_in    / denom_G
        y_N2_out_true  = self.y_N2_in    / denom_G

        C_H2O_out  = float(CH_ss[0])
        x_H2O_out  = max(0.0, min(C_H2O_out / max(geo['c_L_tot'], 1e-9), 1.0))
        x_H2SO4_out = 1.0 - x_H2O_out
        denom_w    = x_H2SO4_out * M_H2SO4 + x_H2O_out * M_H2O
        w_H2SO4_out = (x_H2SO4_out * M_H2SO4 / denom_w
                       if denom_w > 1e-9 else self.w_H2SO4_in)

        return {
            'z': z_arr, 'CG': CG_ss, 'CH': CH_ss,
            'y': y_ss,  'TG': TG_ss, 'TL': TL_ss,
            'y_out': y_out,
            'TG_out': float(TG_ss[-1]),
            'TL_out': float(TL_ss[0]),
            'y_SO3_out': y_SO3_out_true,
            'y_SO2_out': y_SO2_out_true,
            'y_O2_out':  y_O2_out_true,
            'y_N2_out':  y_N2_out_true,
            'C_H2O_out': C_H2O_out,
            'x_H2O_in':  geo['x_H2O_in'],
            'x_H2O_out': x_H2O_out,
            'w_H2SO4_out': w_H2SO4_out,
            'ppm_SO3': y_out * 1e6,
            'eff': eff,
            'converged': converged,
        }

    # ------------------------------------------------------------------
    # Point d'entrée principal
    # ------------------------------------------------------------------
    def compute(self):
        """Calcul stationnaire complet : Onda + BVP (Eq.105–109)."""
        geo  = self._compute_geometry()
        coef = self._onda(geo)
        ss   = self._bvp(geo, coef)

        self.T_gas_out   = ss['TG_out']
        self.T_acid_out  = ss['TL_out']
        self.y_SO3_out   = ss['y_out']
        self.C_SO3_out   = ss.get('C_SO3_out', 0.0)
        self.C_H2O_out   = ss['C_H2O_out']
        self.w_H2SO4_out = ss['w_H2SO4_out']
        self.eff_abs     = ss['eff']
        self.ppm_SO3_out = ss['ppm_SO3']
        self.ss_results  = ss
        return ss


# =====================================================================
# Sous-classes OCP
# =====================================================================
class AbsorptionTowerInter(AbsorptionTower):
    """JD02 — Tour intermédiaire (après lit 3, avant lit 4)."""
    D_COL  = 7.0
    H_PACK = 3.7

    def __init__(self):
        super().__init__(name="Tour d'absorption intermédiaire", tag="JD02")
        self.y_SO3_in   = 0.120
        self.y_SO2_in   = 0.005
        self.y_O2_in    = 0.055
        self.y_N2_in    = 1.0 - self.y_SO3_in - self.y_SO2_in - self.y_O2_in
        self.Q_gas_Nm3h = 118_500.0
        self.Q_acid_m3h = 1_065.0
        self.T_gas_in   = 188.0
        self.T_acid_in  = 80.0
        self.w_H2SO4_in = 0.985
        self.ppm_SO3_out = self.y_SO3_in * 1e6


class AbsorptionTowerFinal(AbsorptionTower):
    """JD03 — Tour finale (après lit 4)."""
    D_COL  = 7.5
    H_PACK = 4.3

    def __init__(self):
        super().__init__(name="Tour d'absorption finale", tag="JD03")
        self.y_SO3_in   = 0.045
        self.y_SO2_in   = 0.002
        self.y_O2_in    = 0.060
        self.y_N2_in    = 1.0 - self.y_SO3_in - self.y_SO2_in - self.y_O2_in
        self.Q_gas_Nm3h = 95_000.0
        self.Q_acid_m3h = 850.0
        self.T_gas_in   = 175.0
        self.T_acid_in  = 80.0
        self.w_H2SO4_in = 0.985
        self.ppm_SO3_out = self.y_SO3_in * 1e6