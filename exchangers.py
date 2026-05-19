from __future__ import annotations
import math

# =============================================================================
# Stubs inline : Equipment, Stream, mixture_properties
# =============================================================================

class Stream:
    """Flux de matiere / energie entre equipements."""
    def __init__(self, name: str = ""):
        self.name = name
        self.temperature: float = 298.15   # K
        self.pressure:    float = 101_325.0 # Pa
        self.flow_rate:   float = 0.0       # kg/s
        self.composition: dict[str, float] = {}

    def __repr__(self) -> str:
        return (
            f"Stream({self.name!r}, T={self.temperature:.1f} K, "
            f"P={self.pressure:.0f} Pa, mdot={self.flow_rate:.3f} kg/s)"
        )


class Equipment:
    """Classe de base pour tous les equipements du simulateur."""
    def __init__(self, name: str, tag: str):
        self.name = name
        self.tag  = tag
        self.input_streams:  dict[str, Stream] = {}
        self.output_streams: dict[str, Stream] = {}
        self.state:   dict = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Interface obligatoire (a surcharger)
    # ------------------------------------------------------------------
    def compute(self, dt: float | None = None) -> dict:
        raise NotImplementedError

    def step(self, dt: float, time: float) -> dict:
        res = self.compute(dt=dt)
        self.log_state(time)
        return res

    def reset(self):
        self.state   = {}
        self.history = []

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    def log_state(self, time: float):
        entry = {"time": time}
        entry.update(self.state)
        self.history.append(entry)


# ── Proprietes thermophysiques des melanges gazeux (O2, N2, SO2, SO3) ─────────
# Donnees de reference a 300 K
_SPECIES_DATA: dict[str, dict[str, float]] = {
    # cp [J/(kg*K)], mu [Pa.s], k [W/(m*K)], M [kg/mol]
    "O2":  {"cp": 920.0,  "mu": 2.06e-5, "k": 0.0263, "M": 0.032},
    "N2":  {"cp": 1040.0, "mu": 1.79e-5, "k": 0.0260, "M": 0.028},
    "SO2": {"cp": 640.0,  "mu": 1.30e-5, "k": 0.0100, "M": 0.064},
    "SO3": {"cp": 635.0,  "mu": 1.40e-5, "k": 0.0085, "M": 0.080},
}

# Exposants de loi de puissance par rapport a T_ref = 300 K
_MU_EXP = {"O2": 0.70, "N2": 0.67, "SO2": 0.82, "SO3": 0.80}
_K_EXP  = {"O2": 0.82, "N2": 0.75, "SO2": 0.90, "SO3": 0.88}
_CP_EXP = {"O2": 0.10, "N2": 0.07, "SO2": 0.20, "SO3": 0.22}
_T_REF  = 300.0  # K


def mixture_properties(
    T_K: float,
    composition: dict[str, float],
    P_Pa: float,
) -> tuple[float, float, float, float, float]:
    """
    Proprietes du melange gazeux a T [K] et P [Pa].

    Retour : (rho [kg/m3], cp [J/(kg*K)], mu [Pa.s], k [W/(m*K)], MW [kg/mol])
    """
    T = max(T_K, 200.0)
    t_ratio = T / _T_REF

    # Normalisation des fractions molaires
    total = sum(max(y, 0.0) for y in composition.values())
    if total <= 0:
        total = 1.0
    y = {sp: max(composition.get(sp, 0.0), 0.0) / total for sp in _SPECIES_DATA}

    # Masse molaire moyenne [kg/mol]
    MW = sum(y[sp] * _SPECIES_DATA[sp]["M"] for sp in _SPECIES_DATA)
    MW = max(MW, 0.020)

    # Masse volumique (gaz parfait)
    R_univ = 8.314462618  # J/(mol*K)
    rho = P_Pa * MW / (R_univ * T)

    # Cp massique moyen [J/(kg*K)]
    cp_mix_molar = sum(
        y[sp] * _SPECIES_DATA[sp]["cp"] * t_ratio ** _CP_EXP.get(sp, 0.1)
        * _SPECIES_DATA[sp]["M"]
        for sp in _SPECIES_DATA
    )
    cp_mix = cp_mix_molar / MW

    # Viscosite dynamique [Pa.s] — loi de melange de Wilke simplifiee
    mu_sp = {sp: _SPECIES_DATA[sp]["mu"] * t_ratio ** _MU_EXP.get(sp, 0.7)
             for sp in _SPECIES_DATA}
    mu_mix   = sum(y[sp] * mu_sp[sp] * math.sqrt(_SPECIES_DATA[sp]["M"])
                   for sp in _SPECIES_DATA)
    denom_mu = sum(y[sp] * math.sqrt(_SPECIES_DATA[sp]["M"]) for sp in _SPECIES_DATA)
    mu = mu_mix / max(denom_mu, 1e-12)

    # Conductivite thermique [W/(m*K)]
    k_sp  = {sp: _SPECIES_DATA[sp]["k"] * t_ratio ** _K_EXP.get(sp, 0.8)
             for sp in _SPECIES_DATA}
    k_mix = sum(y[sp] * k_sp[sp] * math.sqrt(_SPECIES_DATA[sp]["M"])
                for sp in _SPECIES_DATA)
    k = k_mix / max(denom_mu, 1e-12)

    return (
        max(rho,    0.01),
        max(cp_mix, 400.0),
        max(mu,     5e-6),
        max(k,      0.010),
        MW,
    )


# =============================================================================
# HeatExchanger — modele principal
# =============================================================================

class HeatExchanger(Equipment):
    """Echangeur shell-and-tube generique — regime permanent NTU-effectivite."""

    def __init__(
        self,
        name:   str   = "Echangeur de chaleur",
        tag:    str   = "HX",
        # Geometrie tubes
        n_tubes:    int   = 500,
        L_tube:     float = 6.0,    # m
        d_tube_i:   float = 0.040,  # m
        d_tube_o:   float = 0.048,  # m
        k_tube:     float = 45.0,   # W/(m*K) acier au carbone
        # Fluide froid (calandre)
        T_cold_in:      float = 230.0,   # degC
        Q_cold_m3h:     float = 50.0,    # m3/h
        rho_cold:       float = 830.0,   # kg/m3
        cp_cold:        float = 4800.0,  # J/(kg*K)
        h_cold:         float = 3000.0,  # W/(m2*K)
        P_cold_in_kPa:  float = 6200.0,  # kPa
        dp_cold_frac:   float = 0.008,   # perte relative cote utilite
        cold_phase:     str   = "fixed", # fixed | liquid | steam
        # Pertes thermiques
        fouling_factor: float = 2.0e-4,  # m2*K/W
        eta_hx:         float = 0.98,    # rendement
        dp_frac:        float = 0.005,   # perte de charge relative gaz
    ):
        super().__init__(name, tag)

        # Geometrie
        self.n_tubes  = n_tubes
        self.L_tube   = L_tube
        self.d_tube_i = d_tube_i
        self.d_tube_o = d_tube_o
        self.k_tube   = k_tube

        # Fluide froid
        self.T_cold_in     = T_cold_in
        self.Q_cold_m3h    = Q_cold_m3h
        self.rho_cold      = rho_cold
        self.cp_cold       = cp_cold
        self.h_cold        = h_cold
        self.P_cold_in_kPa = P_cold_in_kPa
        self.dp_cold_frac  = dp_cold_frac
        self.cold_phase    = cold_phase

        self._rho_cold_ref   = rho_cold
        self._T_cold_ref     = T_cold_in
        self._P_cold_ref_kPa = P_cold_in_kPa

        # Performance
        self.fouling_factor = fouling_factor
        self.eta_hx         = eta_hx
        self.dp_frac        = dp_frac

        # Modele thermique
        self.model_mode = "thesis_dynamic"
        self.T_amb_C    = 35.0
        self.h_amb      = 12.0   # convection naturelle externe [W/m2/K]
        self.lambda_shell            = 45.0
        self.shell_wall_thickness_m  = 0.02
        self.use_shell_htc_correlation = True

        # Entrees gaz
        self.T_gas_in       = 600.0    # degC
        self.P_gas_in       = 150_000.0 # Pa
        self.gas_composition: dict[str, float] = {}
        self.mdot_gas       = 0.0      # kg/s

        # Resultats
        self.T_gas_out    = self.T_gas_in
        self.T_cold_out   = self.T_cold_in
        self.T_gas_target_out = None
        self.bypass_fraction  = 0.0
        self.Q_exchanged_MW   = 0.0
        self.UA_kW_K          = 0.0
        self.NTU              = 0.0
        self.effectiveness    = 0.0
        self.h_gas            = 0.0
        self.U_overall        = 0.0
        self.Re_gas           = 0.0
        self.h_shell          = h_cold
        self.Re_shell         = 0.0
        self.Pr_shell         = 0.0
        self.Nu_shell         = 0.0
        self.U_loss           = 0.0
        self.A_loss           = 0.0
        self.Q_loss_MW        = 0.0

        # Etats dynamiques
        self.T_tube_out_dyn  = self.T_gas_out
        self.T_shell_out_dyn = self.T_cold_out
        self._init_streams()

    def _init_streams(self):
        self.input_streams["gas_in"]   = Stream("Gaz chaud entree")
        self.output_streams["gas_out"] = Stream("Gaz refroidi sortie")

    # ── Surface d'echange ──────────────────────────────────────────────
    def _A_exchange_inner(self) -> float:
        return self.n_tubes * math.pi * self.d_tube_i * self.L_tube

    def _A_exchange_outer(self) -> float:
        return self.n_tubes * math.pi * self.d_tube_o * self.L_tube

    def _shell_inner_radius(self) -> float:
        a_tubes     = self.n_tubes * math.pi * self.d_tube_o**2 / 4.0
        packing     = 0.60
        a_shell_inner = max(a_tubes / packing, 1e-6)
        return max(math.sqrt(a_shell_inner / math.pi), 0.05)

    def _shell_outer_radius(self) -> float:
        return self._shell_inner_radius() + max(float(self.shell_wall_thickness_m), 1e-4)

    def _shell_length(self) -> float:
        return max(float(self.L_tube), 0.1)

    def _A_shell_inner(self) -> float:
        return 2.0 * math.pi * self._shell_inner_radius() * self._shell_length()

    def _A_shell_outer(self) -> float:
        return 2.0 * math.pi * self._shell_outer_radius() * self._shell_length()

    def _shell_cross_area(self) -> float:
        r_si   = self._shell_inner_radius()
        a_shell   = math.pi * r_si**2
        a_blocked = self.n_tubes * math.pi * self.d_tube_o**2 / 4.0
        return max(a_shell - a_blocked, 1e-4)

    # ── Proprietes transport fluide froid ──────────────────────────────
    def _estimate_cold_transport_props(self) -> tuple[float, float]:
        t_c   = float(self.T_cold_in)
        phase = str(self.cold_phase).lower()
        if phase == "steam":
            t_k = max(t_c + 273.15, 250.0)
            mu  = 1.2e-5 * (t_k / 300.0) ** 0.7
            k   = 0.025 + 2.5e-5 * max(t_c, 0.0)
            return max(mu, 5e-6), max(k, 0.015)
        if phase == "liquid":
            mu = 1.0e-3 * math.exp(-0.025 * (t_c - 20.0))
            mu = max(min(mu, 3.0e-3), 1.0e-4)
            k  = 0.62 - 2.0e-4 * (t_c - 20.0)
            k  = max(min(k, 0.68), 0.25)
            return mu, k
        # fixed
        t_k = max(t_c + 273.15, 250.0)
        mu  = 1.8e-5 * (t_k / 300.0) ** 0.7
        k   = 0.026 + 2.0e-5 * max(t_c, 0.0)
        return max(mu, 8e-6), max(k, 0.018)

    def _compute_shell_htc(self) -> tuple[float, float, float, float]:
        rho     = max(float(self.rho_cold), 1e-6)
        cp      = max(float(self.cp_cold), 1.0)
        q_m3_s  = max(float(self.Q_cold_m3h) / 3600.0, 1e-12)
        a_cross = self._shell_cross_area()
        v       = q_m3_s / max(a_cross, 1e-12)
        mu, k   = self._estimate_cold_transport_props()
        d_h = max(
            4.0 * a_cross / (
                math.pi * self.d_tube_o * self.n_tubes
                + 2.0 * math.pi * self._shell_inner_radius()
            ),
            1e-4,
        )
        re = rho * v * d_h / max(mu, 1e-12)
        pr = cp * mu / max(k, 1e-12)
        if re < 2300.0:
            nu = 3.66
        elif re < 4000.0:
            nu_l = 3.66
            nu_t = 0.023 * 4000.0**0.8 * max(pr, 0.1) ** 0.4
            frac = (re - 2300.0) / 1700.0
            nu   = nu_l + (nu_t - nu_l) * frac
        else:
            nu = 0.023 * re**0.8 * max(pr, 0.1) ** 0.4
        h = nu * k / max(d_h, 1e-12)
        return max(h, 5.0), max(re, 0.0), max(pr, 0.0), max(nu, 0.0)

    def _compute_U_lossA(self, h_shell: float) -> float:
        r_i  = self._shell_inner_radius()
        r_o  = self._shell_outer_radius()
        l_sh = self._shell_length()
        a_i  = self._A_shell_inner()
        a_o  = self._A_shell_outer()
        r_conv_in  = 1.0 / max(h_shell * a_i, 1e-12)
        r_cond     = math.log(max(r_o / max(r_i, 1e-12), 1.0000001)) / max(
            2.0 * math.pi * l_sh * max(self.lambda_shell, 1e-6), 1e-12
        )
        r_conv_out = 1.0 / max(float(self.h_amb) * a_o, 1e-12)
        r_tot = r_conv_in + r_cond + r_conv_out
        return 1.0 / max(r_tot, 1e-12)

    # ── Coefficient transfert cote gaz (Dittus-Boelter) ───────────────
    def _compute_gas_htc(self, T_avg_K: float) -> tuple[float, float]:
        comp = {sp: y for sp, y in self.gas_composition.items()
                if sp in ('O2', 'N2', 'SO2', 'SO3') and y > 0}
        if not comp:
            return 50.0, 1e4
        rho, cp_mass, mu, k_gas, _ = mixture_properties(T_avg_K, comp, self.P_gas_in)
        A_cross_tube  = math.pi / 4.0 * self.d_tube_i ** 2
        A_cross_total = max(self.n_tubes * A_cross_tube, 1e-12)
        v_gas = self.mdot_gas / max(rho * A_cross_total, 1e-12)
        Re    = rho * v_gas * self.d_tube_i / max(mu, 1e-12)
        Pr    = cp_mass * mu / max(k_gas, 1e-12)
        if Re < 2300:
            Nu = 3.66
        elif Re < 4000:
            Nu_lam  = 3.66
            Nu_turb = 0.023 * 4000 ** 0.8 * max(Pr, 0.1) ** 0.3
            frac    = (Re - 2300) / 1700
            Nu      = Nu_lam + (Nu_turb - Nu_lam) * frac
        else:
            Nu = 0.023 * Re ** 0.8 * max(Pr, 0.1) ** 0.3
        h_gas = Nu * k_gas / max(self.d_tube_i, 1e-6)
        return h_gas, Re

    # ── Coefficient global U ───────────────────────────────────────────
    def _compute_U(self, h_gas: float, h_shell: float | None = None) -> float:
        d_i = self.d_tube_i
        d_o = self.d_tube_o
        R_gas  = 1.0 / max(h_gas, 1.0)
        R_foul = self.fouling_factor
        if d_o > d_i > 0 and self.k_tube > 0:
            R_tube = d_i * math.log(d_o / d_i) / (2.0 * self.k_tube)
        else:
            R_tube = 0.0
        h_shell_eff = float(self.h_cold if h_shell is None else h_shell)
        R_cold  = d_i / (max(d_o, 1e-6) * max(h_shell_eff, 1.0))
        R_total = R_gas + R_foul + R_tube + R_cold
        return 1.0 / max(R_total, 1e-6)

    # ── Mise a jour cote froid ─────────────────────────────────────────
    def update_cold_side_from_inputs(self):
        phase = str(self.cold_phase).lower()
        t_k   = max(float(self.T_cold_in) + 273.15, 250.0)
        p_pa  = max(float(self.P_cold_in_kPa) * 1000.0, 1.0e4)
        if phase == "steam":
            mw_h2o = 0.01801528
            rho = p_pa * mw_h2o / (8.314462618 * t_k)
            self.rho_cold = max(min(rho, 150.0), 0.1)
            return
        if phase == "liquid":
            temp_factor  = 1.0 - 3.5e-4 * (float(self.T_cold_in) - float(self._T_cold_ref))
            press_factor = 1.0 + 8.0e-7 * (float(self.P_cold_in_kPa) - float(self._P_cold_ref_kPa))
            rho = float(self._rho_cold_ref) * temp_factor * press_factor
            self.rho_cold = max(min(rho, 1200.0), 600.0)
            return

    # ── Modele dynamique inspire these ────────────────────────────────
    def _compute_thesis_dynamic(self, dt: float | None = None) -> dict:
        if self.mdot_gas <= 0:
            self.state = {"error": "no gas flow"}
            return self.state

        t_gi = float(self.T_gas_in)
        t_ci = float(self.T_cold_in)
        self.update_cold_side_from_inputs()

        comp = {sp: y for sp, y in self.gas_composition.items()
                if sp in ("O2", "N2", "SO2", "SO3") and y > 0}
        if not comp:
            comp = {"N2": 0.79, "O2": 0.10, "SO2": 0.10, "SO3": 0.01}

        t_avg_k = (t_gi + t_ci) * 0.5 + 273.15
        rho_hot, cp_hot, _, _, _ = mixture_properties(t_avg_k, comp, self.P_gas_in)
        c_hot  = max(self.mdot_gas * cp_hot, 1e-9)
        mdot_cold = max(self.Q_cold_m3h * self.rho_cold / 3600.0, 1e-9)
        c_cold = max(mdot_cold * self.cp_cold, 1e-9)

        h_hot, re_hot = self._compute_gas_htc(t_avg_k)
        if self.use_shell_htc_correlation:
            h_shell, re_shell, pr_shell, nu_shell = self._compute_shell_htc()
        else:
            h_shell = max(float(self.h_cold), 1.0)
            re_shell = pr_shell = nu_shell = 0.0

        u_ex    = self._compute_U(h_hot, h_shell)
        a_ex    = self._A_exchange_inner()
        uex_aex = max(u_ex * a_ex * self.eta_hx, 0.0)
        uloss_aloss = max(self._compute_U_lossA(h_shell), 0.0)
        a_loss  = self._A_shell_outer()
        u_loss  = uloss_aloss / max(a_loss, 1e-9)

        a11 = 1.0 + uex_aex / c_hot
        a12 = -uex_aex / c_hot
        a21 = -uex_aex / c_cold
        a22 = 1.0 + uex_aex / c_cold + uloss_aloss / c_cold
        b1  = t_gi
        b2  = t_ci + (uloss_aloss / c_cold) * float(self.T_amb_C)

        x_hot  = float(self.T_tube_out_dyn)
        x_cold = float(self.T_shell_out_dyn)
        if not math.isfinite(x_hot):
            x_hot  = t_gi
        if not math.isfinite(x_cold):
            x_cold = t_ci

        if dt is None:
            det = a11 * a22 - a12 * a21
            if abs(det) < 1e-12:
                t_hot_out_raw  = t_gi
                t_cold_out_raw = t_ci
            else:
                t_hot_out_raw  = (b1 * a22 - a12 * b2) / det
                t_cold_out_raw = (a11 * b2 - b1 * a21) / det
        else:
            a_cross_tube = max(self.n_tubes * math.pi * self.d_tube_i**2 / 4.0, 1e-9)
            v_tube       = max(a_cross_tube * self.L_tube, 1e-9)
            q_hot_m3s    = max(self.mdot_gas / max(rho_hot, 1e-9), 1e-9)
            tau_hot      = max(v_tube / q_hot_m3s, 1e-3)
            v_shell      = max(self._shell_cross_area() * self._shell_length(), 1e-9)
            q_cold_m3s   = max(float(self.Q_cold_m3h) / 3600.0, 1e-9)
            tau_cold     = max(v_shell / q_cold_m3s, 1e-3)
            dx_hot  = (b1 - (a11 * x_hot  + a12 * x_cold)) / tau_hot
            dx_cold = (b2 - (a21 * x_hot  + a22 * x_cold)) / tau_cold
            dt_eff  = max(float(dt), 1e-4)
            dt_int  = min(dt_eff, 0.5 * min(tau_hot, tau_cold))
            t_hot_out_raw  = x_hot  + dt_int * dx_hot
            t_cold_out_raw = x_cold + dt_int * dx_cold

        t_hot_out_raw  = max(min(t_hot_out_raw,  t_gi),       t_ci - 120.0)
        t_cold_out_raw = max(min(t_cold_out_raw, t_gi + 40.0), t_ci - 40.0)

        t_hot_out  = t_hot_out_raw
        t_cold_out = t_cold_out_raw
        bypass     = 0.0
        target     = self.T_gas_target_out

        if target is not None and t_gi > t_ci:
            target = float(target)
            if target >= t_gi:
                t_hot_out = t_gi
                bypass    = 1.0
            elif target > t_hot_out_raw:
                denom  = max(t_gi - t_hot_out_raw, 1e-9)
                bypass = max(min((target - t_hot_out_raw) / denom, 1.0), 0.0)
                t_hot_out  = bypass * t_gi + (1.0 - bypass) * t_hot_out_raw
                q_forced   = max(c_hot * (t_gi - t_hot_out), 0.0)
                t_cold_out = t_ci + q_forced / max(c_cold, 1e-9)

        q_ex_w   = max(c_hot * (t_gi - t_hot_out), 0.0)
        q_loss_w = max(uloss_aloss * max(t_cold_out - float(self.T_amb_C), 0.0), 0.0)
        c_min = max(min(c_hot, c_cold), 1e-9)
        c_max = max(max(c_hot, c_cold), 1e-9)
        cr    = c_min / c_max
        ntu   = uex_aex / c_min
        q_max = c_min * max(t_gi - t_ci, 0.0)
        eps   = q_ex_w / max(q_max, 1e-12) if q_max > 0 else 0.0

        p_gas_out = self.P_gas_in * (1.0 - self.dp_frac)

        self.T_tube_out_dyn  = t_hot_out
        self.T_shell_out_dyn = t_cold_out
        self.T_gas_out       = t_hot_out
        self.T_cold_out      = t_cold_out
        self.bypass_fraction = bypass
        self.Q_exchanged_MW  = q_ex_w / 1e6
        self.Q_loss_MW       = q_loss_w / 1e6
        self.UA_kW_K         = uex_aex / 1000.0
        self.NTU             = ntu
        self.effectiveness   = max(min(eps, 1.0), 0.0)
        self.h_gas           = h_hot
        self.U_overall       = u_ex
        self.Re_gas          = re_hot
        self.h_shell         = h_shell
        self.Re_shell        = re_shell
        self.Pr_shell        = pr_shell
        self.Nu_shell        = nu_shell
        self.U_loss          = u_loss
        self.A_loss          = a_loss

        s_go = self.output_streams["gas_out"]
        s_go.temperature = self.T_gas_out + 273.15
        s_go.pressure    = p_gas_out
        s_go.composition = self.gas_composition.copy()
        s_go.flow_rate   = self.input_streams["gas_in"].flow_rate

        d_t1 = t_gi - t_cold_out
        d_t2 = t_hot_out - t_ci
        if d_t1 > 0 and d_t2 > 0 and abs(d_t1 - d_t2) > 0.01:
            lmtd = (d_t1 - d_t2) / math.log(d_t1 / d_t2)
        elif d_t1 > 0 and d_t2 > 0:
            lmtd = 0.5 * (d_t1 + d_t2)
        else:
            lmtd = 0.0

        result = {
            "name":             self.name,
            "tag":              self.tag,
            "model_mode":       str(self.model_mode),
            "T_gas_in_C":       round(t_gi, 2),
            "T_gas_out_C":      round(self.T_gas_out, 2),
            "T_gas_target_C":   round(float(self.T_gas_target_out), 2) if self.T_gas_target_out is not None else None,
            "T_cold_in_C":      round(t_ci, 2),
            "T_cold_out_C":     round(self.T_cold_out, 2),
            "P_cold_in_kPa":    round(float(self.P_cold_in_kPa), 2),
            "P_cold_out_kPa":   round(float(self.P_cold_in_kPa) * (1.0 - float(self.dp_cold_frac)), 2),
            "Q_cold_m3h":       round(float(self.Q_cold_m3h), 3),
            "mdot_cold_kg_s":   round(float(mdot_cold), 4),
            "cold_phase":       str(self.cold_phase),
            "P_gas_in_kPa":     round(self.P_gas_in / 1000.0, 2),
            "P_gas_out_kPa":    round(p_gas_out / 1000.0, 2),
            "Q_exchanged_MW":   round(self.Q_exchanged_MW, 4),
            "Q_loss_MW":        round(self.Q_loss_MW, 4),
            "LMTD_C":           round(lmtd, 2),
            "UA_kW_K":          round(self.UA_kW_K, 2),
            "U_loss_W_m2K":     round(float(self.U_loss), 2),
            "A_loss_m2":        round(float(self.A_loss), 2),
            "NTU":              round(self.NTU, 4),
            "effectiveness_pct": round(self.effectiveness * 100.0, 2),
            "bypass_pct":       round(self.bypass_fraction * 100.0, 2),
            "h_gas_W_m2K":      round(self.h_gas, 2),
            "h_shell_W_m2K":    round(self.h_shell, 2),
            "U_overall_W_m2K":  round(self.U_overall, 2),
            "Re_gas":           round(self.Re_gas, 2),
            "Re_shell":         round(self.Re_shell, 2),
            "Pr_shell":         round(self.Pr_shell, 4),
            "Nu_shell":         round(self.Nu_shell, 2),
            "A_exchange_m2":    round(self._A_exchange_inner(), 2),
            "Cr":               round(cr, 5),
            "C_hot_W_K":        round(c_hot, 2),
            "C_cold_W_K":       round(c_cold, 2),
        }
        self.state = result
        return result

    # ── Calcul principal (NTU, deux iterations) ────────────────────────
    def compute(self, dt: float | None = None) -> dict:
        mode = str(getattr(self, "model_mode", "ntu")).lower()
        if mode in {"thesis_dynamic", "dynamic", "eq114"}:
            return self._compute_thesis_dynamic(dt=dt)

        if self.mdot_gas <= 0:
            self.state = {"error": "no gas flow"}
            return self.state

        T_gi = self.T_gas_in
        T_ci = self.T_cold_in
        self.update_cold_side_from_inputs()

        T_avg_est = (T_gi + T_ci) / 2.0 + 273.15
        comp = {sp: y for sp, y in self.gas_composition.items()
                if sp in ('O2', 'N2', 'SO2', 'SO3') and y > 0}
        if not comp:
            comp = {'N2': 0.79, 'O2': 0.10, 'SO2': 0.10, 'SO3': 0.01}

        _, cp_gas, _, _, _ = mixture_properties(T_avg_est, comp, self.P_gas_in)
        C_hot   = self.mdot_gas * cp_gas
        mdot_cold = self.Q_cold_m3h * self.rho_cold / 3600.0
        C_cold  = mdot_cold * self.cp_cold

        if C_hot <= 0 or C_cold <= 0:
            self.T_gas_out = T_gi
            self.state = {"T_gas_out_C": T_gi, "error": "zero capacity"}
            return self.state

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        Cr    = C_min / C_max

        h_gas, Re = self._compute_gas_htc(T_avg_est)
        if self.use_shell_htc_correlation:
            h_shell, re_shell, pr_shell, nu_shell = self._compute_shell_htc()
        else:
            h_shell = max(float(self.h_cold), 1.0)
            re_shell = pr_shell = nu_shell = 0.0

        U   = self._compute_U(h_gas, h_shell)
        A_i = self._A_exchange_inner()
        UA  = U * A_i * self.eta_hx
        NTU = UA / C_min if C_min > 0 else 0.0

        if Cr < 0.999:
            exp_term = math.exp(-NTU * (1.0 - Cr))
            denom_e  = 1.0 - Cr * exp_term
            eps = (1.0 - exp_term) / denom_e if denom_e != 0 else 0.0
        else:
            eps = NTU / (1.0 + NTU)
        eps = max(0.0, min(eps, 1.0))

        Q_max    = C_min * (T_gi - T_ci)
        Q        = eps * Q_max
        T_gas_out  = T_gi - Q / max(C_hot, 1e-6)
        T_cold_out = T_ci + Q / max(C_cold, 1e-6)

        # Iteration 2 avec temperature moyenne reelle
        T_avg_real = (T_gi + T_gas_out) / 2.0 + 273.15
        _, cp_gas2, _, _, _ = mixture_properties(T_avg_real, comp, self.P_gas_in)
        C_hot2 = self.mdot_gas * cp_gas2
        C_min2 = min(C_hot2, C_cold)
        C_max2 = max(C_hot2, C_cold)
        Cr2    = C_min2 / C_max2
        h_gas2, Re2 = self._compute_gas_htc(T_avg_real)
        U2   = self._compute_U(h_gas2, h_shell)
        UA2  = U2 * A_i * self.eta_hx
        NTU2 = UA2 / C_min2 if C_min2 > 0 else 0.0

        if Cr2 < 0.999:
            exp_term2 = math.exp(-NTU2 * (1.0 - Cr2))
            denom_e2  = 1.0 - Cr2 * exp_term2
            eps2 = (1.0 - exp_term2) / denom_e2 if denom_e2 != 0 else 0.0
        else:
            eps2 = NTU2 / (1.0 + NTU2)
        eps2 = max(0.0, min(eps2, 1.0))

        Q_max2          = C_min2 * (T_gi - T_ci)
        Q2              = eps2 * Q_max2
        T_gas_out_raw   = T_gi - Q2 / max(C_hot2, 1e-6)
        T_cold_out_raw  = T_ci + Q2 / max(C_cold, 1e-6)

        T_gas_out  = T_gas_out_raw
        T_cold_out = T_cold_out_raw
        bypass     = 0.0
        Q_actual   = Q2
        target     = self.T_gas_target_out

        if target is not None:
            target = float(target)
            if T_gi > T_ci:
                if target >= T_gi:
                    T_gas_out = T_gi
                    bypass    = 1.0
                elif target > T_gas_out_raw:
                    denom_bypass = max(T_gi - T_gas_out_raw, 1e-9)
                    bypass = (target - T_gas_out_raw) / denom_bypass
                    bypass = max(0.0, min(bypass, 1.0))
                    T_gas_out = bypass * T_gi + (1.0 - bypass) * T_gas_out_raw
                    Q_actual  = max(C_hot2 * (T_gi - T_gas_out), 0.0)
                    Q_actual  = min(Q_actual, max(Q2, 0.0))
                    T_cold_out = T_ci + Q_actual / max(C_cold, 1e-6)
                    eps2 = Q_actual / max(Q_max2, 1e-12) if Q_max2 > 0 else 0.0

        self.T_gas_out       = T_gas_out
        self.T_cold_out      = T_cold_out
        self.bypass_fraction = bypass
        self.Q_exchanged_MW  = Q_actual / 1.0e6
        self.UA_kW_K         = UA2 / 1000.0
        self.NTU             = NTU2
        self.effectiveness   = max(0.0, min(eps2, 1.0))
        self.h_gas           = h_gas2
        self.U_overall       = U2
        self.Re_gas          = Re2
        self.h_shell         = h_shell
        self.Re_shell        = re_shell
        self.Pr_shell        = pr_shell
        self.Nu_shell        = nu_shell
        self.U_loss          = 0.0
        self.A_loss          = self._A_shell_outer()
        self.Q_loss_MW       = 0.0
        self.T_tube_out_dyn  = self.T_gas_out
        self.T_shell_out_dyn = self.T_cold_out

        P_gas_out = self.P_gas_in * (1.0 - self.dp_frac)
        s_go = self.output_streams["gas_out"]
        s_go.temperature = self.T_gas_out + 273.15
        s_go.pressure    = P_gas_out
        s_go.composition = self.gas_composition.copy()
        s_go.flow_rate   = self.input_streams["gas_in"].flow_rate

        dT1 = T_gi - T_cold_out
        dT2 = T_gas_out - T_ci
        if dT1 > 0 and dT2 > 0 and abs(dT1 - dT2) > 0.01:
            LMTD = (dT1 - dT2) / math.log(dT1 / dT2)
        elif dT1 > 0 and dT2 > 0:
            LMTD = (dT1 + dT2) / 2.0
        else:
            LMTD = 0.0

        result = {
            "name":             self.name,
            "tag":              self.tag,
            "model_mode":       str(self.model_mode),
            "T_gas_in_C":       round(T_gi, 2),
            "T_gas_out_C":      round(self.T_gas_out, 2),
            "T_gas_target_C":   round(float(self.T_gas_target_out), 2) if self.T_gas_target_out is not None else None,
            "T_cold_in_C":      round(T_ci, 2),
            "T_cold_out_C":     round(T_cold_out, 2),
            "P_cold_in_kPa":    round(float(self.P_cold_in_kPa), 2),
            "P_cold_out_kPa":   round(float(self.P_cold_in_kPa) * (1.0 - float(self.dp_cold_frac)), 2),
            "Q_cold_m3h":       round(float(self.Q_cold_m3h), 3),
            "mdot_cold_kg_s":   round(float(mdot_cold), 4),
            "cold_phase":       str(self.cold_phase),
            "P_gas_in_kPa":     round(self.P_gas_in / 1000.0, 2),
            "P_gas_out_kPa":    round(P_gas_out / 1000.0, 2),
            "Q_exchanged_MW":   round(self.Q_exchanged_MW, 4),
            "Q_loss_MW":        round(self.Q_loss_MW, 4),
            "LMTD_C":           round(LMTD, 1),
            "UA_kW_K":          round(self.UA_kW_K, 1),
            "U_loss_W_m2K":     round(float(self.U_loss), 2),
            "A_loss_m2":        round(float(self.A_loss), 2),
            "NTU":              round(self.NTU, 3),
            "effectiveness_pct": round(self.effectiveness * 100.0, 1),
            "bypass_pct":       round(self.bypass_fraction * 100.0, 2),
            "h_gas_W_m2K":      round(self.h_gas, 1),
            "h_shell_W_m2K":    round(self.h_shell, 2),
            "U_overall_W_m2K":  round(self.U_overall, 1),
            "Re_gas":           round(self.Re_gas, 0),
            "Re_shell":         round(self.Re_shell, 2),
            "Pr_shell":         round(self.Pr_shell, 4),
            "Nu_shell":         round(self.Nu_shell, 2),
            "A_exchange_m2":    round(self._A_exchange_inner(), 1),
            "Cr":               round(Cr2, 4),
            "C_hot_W_K":        round(C_hot2, 1),
            "C_cold_W_K":       round(C_cold, 1),
        }
        self.state = result
        return result

    def set_gas_inlet(
        self,
        T_C: float,
        P_Pa: float,
        mdot_kg_s: float,
        composition: dict[str, float],
    ):
        """Configure les conditions d'entree du gaz."""
        self.T_gas_in       = T_C
        self.P_gas_in       = P_Pa
        self.mdot_gas       = mdot_kg_s
        self.gas_composition = composition.copy()

    def step(self, dt: float, time: float) -> dict:
        res = self.compute(dt=dt)
        self.log_state(time)
        return res

    def reset(self):
        self.T_gas_out       = self.T_gas_in
        self.T_cold_out      = self.T_cold_in
        self.T_tube_out_dyn  = self.T_gas_in
        self.T_shell_out_dyn = self.T_cold_in
        self.T_gas_target_out = None
        self.bypass_fraction  = 0.0
        self.Q_exchanged_MW   = 0.0
        self.Q_loss_MW        = 0.0
        self.UA_kW_K          = 0.0
        self.NTU              = 0.0
        self.effectiveness    = 0.0
        self.h_shell          = self.h_cold
        self.Re_shell         = 0.0
        self.Pr_shell         = 0.0
        self.Nu_shell         = 0.0
        self.U_loss           = 0.0
        self.A_loss           = 0.0
        self.history          = []
        self.state            = {}


# =============================================================================
# Sous-classes preconfigurees pour le train interpass OCP Jorf Lasfar
# =============================================================================

class HPSuperheater1B(HeatExchanger):
    """
    HP Superheater 1B (401AE05) — entre lit 1 (F10) et lit 2 (F11).
    Gaz : F10=627.08 degC -> F11=440 degC
    Cote calandre : vapeur HP saturee F121=263.71 degC / 6123 kPa
    """
    F121_T_cold_in = 243.2
    F120_P_kPa     = 6188

    def __init__(self):
        super().__init__(
            name="HP Superheater 1B",
            tag="401AE05",
            n_tubes=2200,
            L_tube=10.0,
            d_tube_i=0.044,
            d_tube_o=0.051,
            k_tube=45.0,
            T_cold_in=263.71,
            Q_cold_m3h=217.95,
            rho_cold=37.0,
            cp_cold=2800.0,
            h_cold=1500.0,
            P_cold_in_kPa=6123.0,
            dp_cold_frac=0.006,
            cold_phase="steam",
            fouling_factor=2.0e-4,
            eta_hx=0.97,
            dp_frac=0.003,
        )


class HotInterpassHX(HeatExchanger):
    """
    Hot Interpass Heat Exchanger (401AE19) — entre lit 2 (F12) et lit 3 (F13).
    Gaz : ~516 degC -> ~440 degC
    Cote calandre : gaz recycle JD02 (~107 degC -> ~300 degC)
    """
    def __init__(self):
        super().__init__(
            name="Hot Interpass HX",
            tag="401AE19",
            n_tubes=1800,
            L_tube=8.0,
            d_tube_i=0.044,
            d_tube_o=0.051,
            k_tube=45.0,
            T_cold_in=107.0,
            Q_cold_m3h=98000.0,
            rho_cold=1.8,
            cp_cold=1050.0,
            h_cold=80.0,
            P_cold_in_kPa=128.5,
            dp_cold_frac=0.008,
            cold_phase="fixed",
            fouling_factor=1.5e-4,
            eta_hx=0.97,
            dp_frac=0.003,
        )


class ColdInterpassHX(HeatExchanger):
    """
    Cold Interpass Heat Exchanger (401AE18) — apres lit 3 (F14 -> F15).
    Gaz : ~460 degC -> ~306 degC
    Cote calandre : gaz recycle JD02 (~300 degC -> ~393 degC)
    """
    def __init__(self):
        super().__init__(
            name="Cold Interpass HX",
            tag="401AE18",
            n_tubes=1800,
            L_tube=8.0,
            d_tube_i=0.044,
            d_tube_o=0.051,
            k_tube=45.0,
            T_cold_in=300.0,
            Q_cold_m3h=98000.0,
            rho_cold=0.95,
            cp_cold=1080.0,
            h_cold=85.0,
            P_cold_in_kPa=128.5,
            dp_cold_frac=0.008,
            cold_phase="fixed",
            fouling_factor=1.5e-4,
            eta_hx=0.97,
            dp_frac=0.003,
        )


class Economizer3B(HeatExchanger):
    """
    Economizer 3B (401AE10) — F15 -> F16 (entree JD02).
    Gaz : ~306 degC -> ~179 degC
    Cote calandre : vapeur utilite
    """
    def __init__(self):
        super().__init__(
            name="Economizer 3B",
            tag="401AE10",
            n_tubes=1500,
            L_tube=8.0,
            d_tube_i=0.038,
            d_tube_o=0.044,
            k_tube=45.0,
            T_cold_in=130.0,
            Q_cold_m3h=136.0,
            rho_cold=32.0,
            cp_cold=2900.0,
            h_cold=1500.0,
            P_cold_in_kPa=6711.0,
            dp_cold_frac=0.006,
            cold_phase="steam",
            fouling_factor=2.0e-4,
            eta_hx=0.98,
            dp_frac=0.003,
        )


class FinalCoolingTrain(HeatExchanger):
    """
    Train de refroidissement final equivalent : HP4A + LP4A + E4C + E4A.
    Gaz : ~406 degC -> ~134 degC
    """
    def __init__(self):
        super().__init__(
            name="HP4A + LP4A + E4C + E4A",
            tag="FINAL-COOL",
            n_tubes=3500,
            L_tube=8.0,
            d_tube_i=0.040,
            d_tube_o=0.048,
            k_tube=45.0,
            T_cold_in=105.0,
            Q_cold_m3h=323.0,
            rho_cold=960.0,
            cp_cold=4200.0,
            h_cold=3000.0,
            P_cold_in_kPa=6819.0,
            dp_cold_frac=0.008,
            cold_phase="liquid",
            fouling_factor=2.5e-4,
            eta_hx=0.96,
            dp_frac=0.005,
        )


class HP4AExchanger(HeatExchanger):
    """HP 4A superheater stage (final train, after bed 4 gas). Utility: HP steam."""
    def __init__(self):
        super().__init__(
            name="HP 4A",
            tag="HP4A",
            n_tubes=900,
            L_tube=8.0,
            d_tube_i=0.040,
            d_tube_o=0.048,
            k_tube=45.0,
            T_cold_in=184.3,
            Q_cold_m3h=220.0,
            rho_cold=35.0,
            cp_cold=2900.0,
            h_cold=1600.0,
            P_cold_in_kPa=6407.0,
            dp_cold_frac=0.006,
            cold_phase="steam",
            fouling_factor=2.0e-4,
            eta_hx=0.97,
            dp_frac=0.0012,
        )


class LP4AExchanger(HeatExchanger):
    """LP 4A superheater stage. Utility: LP steam."""
    def __init__(self):
        super().__init__(
            name="LP 4A",
            tag="LP4A",
            n_tubes=850,
            L_tube=8.0,
            d_tube_i=0.040,
            d_tube_o=0.048,
            k_tube=45.0,
            T_cold_in=168.0,
            Q_cold_m3h=180.0,
            rho_cold=4.5,
            cp_cold=2500.0,
            h_cold=1300.0,
            P_cold_in_kPa=788.3,
            dp_cold_frac=0.008,
            cold_phase="steam",
            fouling_factor=2.0e-4,
            eta_hx=0.97,
            dp_frac=0.0012,
        )


class E4CExchanger(HeatExchanger):
    """E 4C economizer stage. Utility: feed water."""
    def __init__(self):
        super().__init__(
            name="E 4C",
            tag="E4C",
            n_tubes=900,
            L_tube=8.0,
            d_tube_i=0.039,
            d_tube_o=0.046,
            k_tube=45.0,
            T_cold_in=212.5,
            Q_cold_m3h=145.0,
            rho_cold=900.0,
            cp_cold=4300.0,
            h_cold=3200.0,
            P_cold_in_kPa=6613.0,
            dp_cold_frac=0.006,
            cold_phase="liquid",
            fouling_factor=2.0e-4,
            eta_hx=0.97,
            dp_frac=0.0012,
        )


class E4AExchanger(HeatExchanger):
    """E 4A economizer stage (last stage before final absorption). Utility: HP BFW."""
    def __init__(self):
        super().__init__(
            name="E 4A",
            tag="E4A",
            n_tubes=1100,
            L_tube=8.0,
            d_tube_i=0.039,
            d_tube_o=0.046,
            k_tube=45.0,
            T_cold_in=107.3,
            Q_cold_m3h=170.0,
            rho_cold=945.0,
            cp_cold=4200.0,
            h_cold=3300.0,
            P_cold_in_kPa=6819.0,
            dp_cold_frac=0.006,
            cold_phase="liquid",
            fouling_factor=2.0e-4,
            eta_hx=0.97,
            dp_frac=0.0014,
        )