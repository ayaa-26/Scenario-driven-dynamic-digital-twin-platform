#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbine à vapeur — modèle simple pour entraînement mécanique (arbre).
Objectif
--------
Fournir une source de puissance mécanique (kW) à partir d'une détente vapeur :
- entrée : P_in, T_in, ouverture vanne (0–1) => débit massique vapeur
- sortie : P_out, T_out
- puissance : W_shaft_kW (après rendements isentropique + mécanique)
"""

from __future__ import annotations
import numpy as np

# Constantes physiques intégrées directement
R_CONST = 8.314        # J/(mol·K)
M_H2O   = 0.018015     # kg/mol


class SteamTurbine:
    """Turbine vapeur (modèle gaz parfait) pour entraînement d'une charge."""

    GAMMA     = 1.30      # -
    CP_STEAM  = 2_100.0   # J/(kg·K)

    def __init__(self, name: str = "Turbine vapeur", tag: str = "ST-DRIVE"):
        self.name = name
        self.tag  = tag

        # Conditions vapeur — calées sur DCS OCP Jorf Lasfar
        self.P_in   = 62.6e5   # Pa (~63 bar HP)
        self.T_in   = 400.0    # °C (vapeur surchauffée)
        self.P_out  = 13.51e5  # Pa (~13.5 bar MP)

        # Commande vanne
        self.valve     = 0.70   # 0–1
        self.mdot_max  = 25.0   # kg/s

        # Rendements
        self.eta_is   = 0.78
        self.eta_mech = 0.98

        # Résultats
        self.mdot_kg_s  = 0.0
        self.T_out_calc = self.T_in
        self.W_shaft_kW = 0.0
        self.state      = {}
        self.history    = []

    # ------------------------------------------------------------------
    def _specific_work_J_kg(self) -> float:
        P_in  = float(self.P_in)
        P_out = float(self.P_out)
        if P_in <= 0.0 or P_out <= 0.0 or P_out >= P_in:
            return 0.0
        T_in_K = float(self.T_in) + 273.15
        exp    = (self.GAMMA - 1.0) / self.GAMMA
        T_is_K = T_in_K * (P_out / P_in) ** exp
        dh_is  = self.CP_STEAM * max(T_in_K - T_is_K, 0.0)
        return float(self.eta_mech * self.eta_is * dh_is)

    def required_mdot_for_power(self, W_shaft_kW: float) -> float:
        w = self._specific_work_J_kg()
        if w <= 0.0:
            return float("inf")
        return max(float(W_shaft_kW), 0.0) * 1_000.0 / w

    def valve_for_power(self, W_shaft_kW: float) -> float:
        mdot = self.required_mdot_for_power(W_shaft_kW)
        if not np.isfinite(mdot) or self.mdot_max <= 0.0:
            return 0.0
        return float(np.clip(mdot / self.mdot_max, 0.0, 1.0))

    # ------------------------------------------------------------------
    def compute(self, valve: float | None = None) -> dict:
        if valve is not None:
            self.valve = float(np.clip(valve, 0.0, 1.0))

        self.mdot_kg_s = max(self.valve, 0.0) * max(self.mdot_max, 0.0)

        P_in  = float(self.P_in)
        P_out = float(self.P_out)
        T_in_K = float(self.T_in) + 273.15

        if self.mdot_kg_s <= 0.0 or P_in <= 0.0 or P_out <= 0.0 or P_out >= P_in:
            self.T_out_calc = float(self.T_in)
            self.W_shaft_kW = 0.0
        else:
            exp    = (self.GAMMA - 1.0) / self.GAMMA
            T_is_K = T_in_K * (P_out / P_in) ** exp
            T_out_K = T_in_K - self.eta_is * max(T_in_K - T_is_K, 0.0)
            self.T_out_calc = T_out_K - 273.15
            w_shaft = self._specific_work_J_kg()
            self.W_shaft_kW = self.mdot_kg_s * w_shaft / 1_000.0

        result = {
            "valve_pct"         : self.valve * 100.0,
            "mdot_kg_s"         : self.mdot_kg_s,
            "P_in_Pa"           : float(self.P_in),
            "P_out_Pa"          : float(self.P_out),
            "T_in_C"            : float(self.T_in),
            "T_out_C"           : float(self.T_out_calc),
            "W_shaft_kW"        : float(self.W_shaft_kW),
            "w_shaft_kJ_kg"     : self._specific_work_J_kg() / 1_000.0,
        }
        self.state = result
        return result

    def step(self, dt: float, time: float) -> dict:
        res = self.compute(self.valve)
        self.history.append({"time": time, **res})
        return res

    def reset(self):
        self.valve      = 0.70
        self.mdot_kg_s  = 0.0
        self.T_out_calc = self.T_in
        self.W_shaft_kW = 0.0
        self.history    = []
        self.state      = {}