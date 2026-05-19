#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbosoufflante centrifuge JC01 — Jumeau Numérique OCP
Unité acide sulfurique
"""

from __future__ import annotations
import numpy as np

# Constantes physiques intégrées directement
P_ATM   = 101_325.0   # Pa
R_CONST = 8.314        # J/(mol·K)
M_O2    = 0.032        # kg/mol
M_N2    = 0.028        # kg/mol
M_H2O   = 0.018015     # kg/mol


class TurboBlower:
    """
    Turbosoufflante centrifuge JC01.

    Point nominal OCP — bilan gazeux streams 3 → 4
    -----------------------------------------------
    Q_nom  = 363 934 Nm³/h
    ΔP_nom = 5 769 mm W.C. = 56 594 Pa
    T_in   = 66 °C  |  T_out = 123 °C  |  r_c = 1.582
    η_is   = 0.83
    """

    Q_NOM      = 363_934.0
    DP_NOM     = 56_594.0
    ETA_IS     = 0.83
    GAMMA      = 1.40
    CP_AIR     = 1_005.0
    SURGE_FRAC = 0.65
    N_NOM      = 4_883.0
    P_IN_NOM   = P_ATM - 412.0 * 9.81   # ≈ 97 283 Pa

    _DP_SHUTOFF = 1.20 * DP_NOM
    _K_CURVE    = (_DP_SHUTOFF - DP_NOM) / Q_NOM ** 2

    def __init__(self, name: str = "Compresseurs principaux", tag: str = "401AC01/02"):
        self.name = name
        self.tag  = tag

        self.Q_gas_Nm3h = self.Q_NOM
        self.T_in       = 66.0
        self.P_in       = self.P_IN_NOM

        self.y_O2  = 76_235.0  / 363_934.0
        self.y_N2  = 287_699.0 / 363_934.0
        self.y_H2O = 0.0

        self.load = 1.0

        self.delta_P    = self.DP_NOM
        self.P_out      = self.P_in + self.delta_P
        self.T_out      = self.T_in
        self.W_shaft_kW = 0.0
        self.r_compress = self.P_out / self.P_in
        self.surge_margin = 100.0
        self.is_surging   = False

        self.state   = {}
        self.history = []

    # ------------------------------------------------------------------
    def _molar_mass_mix(self) -> float:
        return self.y_O2 * M_O2 + self.y_N2 * M_N2 + self.y_H2O * M_H2O

    def _mass_flow_kg_s(self) -> float:
        rho_N = P_ATM * self._molar_mass_mix() / (R_CONST * 273.15)
        return self.Q_gas_Nm3h * rho_N / 3_600.0

    def _characteristic_curve(self) -> float:
        f  = self.load
        dp = f ** 2 * self._DP_SHUTOFF - self._K_CURVE * self.Q_gas_Nm3h ** 2
        return max(dp, 0.0)

    def _isentropic_compression(self, delta_p: float) -> tuple[float, float]:
        T_in_K = self.T_in + 273.15
        P_out  = self.P_in + delta_p
        r_c    = P_out / max(self.P_in, 1.0)
        exp    = (self.GAMMA - 1.0) / self.GAMMA
        T_is_K = T_in_K * (r_c ** exp)
        T_out_K = T_in_K + (T_is_K - T_in_K) / self.ETA_IS
        m_dot  = self._mass_flow_kg_s()
        W_is   = m_dot * self.CP_AIR * (T_is_K - T_in_K)
        W_shaft = W_is / self.ETA_IS
        return T_out_K - 273.15, W_shaft / 1_000.0

    def _surge_check(self) -> tuple[float, bool]:
        Q_surge = self.SURGE_FRAC * self.load * self.Q_NOM
        margin  = (self.Q_gas_Nm3h - Q_surge) / max(Q_surge, 1.0) * 100.0
        return margin, (margin < 0.0)

    def _outlet_density(self) -> float:
        T_out_K = self.T_out + 273.15
        return self.P_out * self._molar_mass_mix() / (R_CONST * T_out_K)

    # ------------------------------------------------------------------
    def set_inlet_conditions(self, T_in_C: float, P_in_Pa: float,
                              Q_Nm3h: float | None = None):
        """Met à jour les conditions d'entrée."""
        self.T_in = float(T_in_C)
        self.P_in = float(P_in_Pa)
        if Q_Nm3h is not None:
            self.Q_gas_Nm3h = float(Q_Nm3h)

    def load_for_delta_p(self, delta_p: float) -> float:
        dp  = max(float(delta_p), 0.0)
        num = dp + self._K_CURVE * self.Q_gas_Nm3h ** 2
        if num <= 0.0 or self._DP_SHUTOFF <= 0.0:
            f = 0.0
        else:
            f = float(np.sqrt(num / self._DP_SHUTOFF))
        return float(np.clip(f, 0.30, 1.10))

    def load_for_P_out(self, P_out: float) -> float:
        return self.load_for_delta_p(float(P_out) - float(self.P_in))

    def _w_shaft_for_load(self, load: float) -> float:
        f  = float(np.clip(load, 0.30, 1.10))
        dp = f ** 2 * self._DP_SHUTOFF - self._K_CURVE * self.Q_gas_Nm3h ** 2
        dp = max(dp, 0.0)
        _, W_kW = self._isentropic_compression(dp)
        return float(W_kW)

    def load_for_shaft_power(self, W_shaft_kW: float, tol_kW: float = 1.0) -> float:
        target = max(float(W_shaft_kW), 0.0)
        lo, hi = 0.30, 1.10
        W_lo = self._w_shaft_for_load(lo)
        W_hi = self._w_shaft_for_load(hi)
        if target <= W_lo:
            return lo
        if target >= W_hi:
            return hi
        grid = np.linspace(lo, hi, 60)
        Wg   = np.array([self._w_shaft_for_load(f) for f in grid], dtype=float)
        idx  = int(np.searchsorted(Wg, target, side="left"))
        idx  = max(1, min(idx, len(grid) - 1))
        lo   = float(grid[idx - 1])
        hi   = float(grid[idx])
        for _ in range(50):
            mid   = 0.5 * (lo + hi)
            W_mid = self._w_shaft_for_load(mid)
            if abs(W_mid - target) <= tol_kW:
                return float(mid)
            if W_mid < target:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    # ------------------------------------------------------------------
    def compute(self, load: float | None = None) -> dict:
        if load is not None:
            self.load = float(np.clip(load, 0.30, 1.10))

        self.delta_P    = self._characteristic_curve()
        self.P_out      = self.P_in + self.delta_P
        self.r_compress = self.P_out / max(self.P_in, 1.0)

        self.T_out, self.W_shaft_kW = self._isentropic_compression(self.delta_P)

        self.surge_margin, self.is_surging = self._surge_check()
        rho_out = self._outlet_density()

        result = {
            "delta_P_Pa"      : self.delta_P,
            "delta_P_mmCE"    : self.delta_P / 9.81,
            "P_in_Pa"         : self.P_in,
            "P_out_Pa"        : self.P_out,
            "P_out_kPa"       : self.P_out / 1_000.0,
            "r_compression"   : self.r_compress,
            "T_in_C"          : self.T_in,
            "T_out_C"         : self.T_out,
            "delta_T_C"       : self.T_out - self.T_in,
            "Q_in_Nm3h"       : self.Q_gas_Nm3h,
            "mdot_kg_s"       : self._mass_flow_kg_s(),
            "W_shaft_kW"      : self.W_shaft_kW,
            "eta_is"          : self.ETA_IS,
            "surge_margin_pct": self.surge_margin,
            "is_surging"      : self.is_surging,
            "load_pct"        : self.load * 100.0,
            "N_rpm"           : self.load * self.N_NOM,
            "N_nom_rpm"       : self.N_NOM,
            "rho_out_kg_m3"   : rho_out,
        }
        self.state = result
        return result

    def step(self, dt: float, time: float) -> dict:
        res = self.compute(self.load)
        self.history.append({"time": time, **res})
        return res

    def reset(self):
        self.load         = 1.0
        self.T_in         = 66.0
        self.P_in         = self.P_IN_NOM
        self.Q_gas_Nm3h   = self.Q_NOM
        self.y_O2         = 76_235.0  / 363_934.0
        self.y_N2         = 287_699.0 / 363_934.0
        self.y_H2O        = 0.0
        self.delta_P      = self.DP_NOM
        self.P_out        = self.P_IN_NOM + self.DP_NOM
        self.T_out        = 66.0
        self.W_shaft_kW   = 0.0
        self.r_compress   = (self.P_IN_NOM + self.DP_NOM) / self.P_IN_NOM
        self.surge_margin = 100.0
        self.is_surging   = False
        self.history      = []
        self.state        = {}