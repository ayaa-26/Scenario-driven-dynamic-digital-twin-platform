#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filtre d'entrée d'air 401AS02 — Jumeau Numérique OCP
Unité acide sulfurique — Tour de séchage T-101

Point de calibration (bilan gazeux OCP) :
  Stream 1 → Stream 2 : ΔP = 76 mm W.C. = 745 Pa
  Q_ref = 368 665 Nm³/h,  T = 30°C,  composition inchangée
"""

import numpy as np

# ── Constantes ──────────────────────────────────────────────────────────────
P_ATM = 101325.0      # Pa
R_CONST = 8.314       # J/(mol·K)
M_O2 = 0.031999       # kg/mol
M_N2 = 0.028014       # kg/mol
M_H2O = 0.018015      # kg/mol


class AirFilter:
    """
    Filtre d'entrée d'air 401AS02.

    Physique implémentée
    --------------------
    - Loi puissance  ΔP = ΔP_ref × (Q/Q_ref)^n      [Perry's, ch.6]
    - Colmatage      ΔP(t) = ΔP_clean × (1 + α × t)
    - Sortie isotherme : T_out = T_in, composition inchangée
    - Densité corrigée à P_out pour alimenter DryingTower

    Point nominal OCP
    -----------------
    Q_ref = 368 665 Nm³/h  →  ΔP_ref = 76 mm W.C. = 745 Pa
    """

    # ── Paramètres filtre ───────────────────────────────────────────────────────
    Q_REF      = 368_665.0   # Nm³/h  — débit nominal OCP
    DP_REF     = 745.0       # Pa     — ΔP nominal (76 mm W.C. × 9.81)
    N_EXPO     = 1.7         # −      — exposant loi puissance (fibres industrielles)
    ALPHA_FOUL = 0.0         # 1/h    — taux de colmatage (0 = filtre propre)

    def __init__(self, name="Filtre d'air entrée", tag="401AS02"):
        self.name = name
        self.tag = tag

        # ── Conditions entrée (Stream 1 — air ambiant) ────────────────────────
        self.Q_gas_Nm3h = 368_665.0
        self.T_in       = 30.0      # °C
        self.P_in       = P_ATM     # Pa

        # ── Composition (fractions molaires — identiques streams 1 et 2) ──────
        self.y_O2  = 76_235.0  / 368_665.0
        self.y_N2  = 287_699.0 / 368_665.0
        self.y_H2O = 4_731.0   / 368_665.0

        # ── Sorties courantes ─────────────────────────────────────────────────
        self.delta_P        = self.DP_REF
        self.P_out          = self.P_in - self.delta_P
        self.T_out          = self.T_in
        self.fouling_factor = 1.0

        self.state = {}
        self.history = []

    # ─────────────────────────────────────────────────────────────────────────
    # Privé
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_delta_p(self, t_hours: float = 0.0) -> float:
        """
        ΔP = ΔP_ref × (Q/Q_ref)^n × (1 + α_foul × t)

        Paramètres
        ----------
        t_hours : temps écoulé depuis dernier changement filtre [h]
        """
        ratio = self.Q_gas_Nm3h / self.Q_REF
        dp_flow = self.DP_REF * (ratio ** self.N_EXPO)
        self.fouling_factor = 1.0 + self.ALPHA_FOUL * t_hours
        return dp_flow * self.fouling_factor

    def _outlet_density(self) -> float:
        """ρ_G à P_out — gaz idéal, loi des gaz parfaits."""
        M_mix = self.y_O2 * M_O2 + self.y_N2 * M_N2 + self.y_H2O * M_H2O
        T_K   = self.T_out + 273.15
        return self.P_out * M_mix / (R_CONST * T_K)

    # ─────────────────────────────────────────────────────────────────────────
    # Interface publique
    # ─────────────────────────────────────────────────────────────────────────

    def compute(self, t_hours: float = 0.0) -> dict:
        """Calcul stationnaire du filtre."""
        self.delta_P = self._compute_delta_p(t_hours)
        self.P_out   = self.P_in - self.delta_P
        self.T_out   = self.T_in          # filtre isotherme

        rho_out = self._outlet_density()

        result = {
            'delta_P_Pa':     self.delta_P,
            'delta_P_mmWC':   self.delta_P / 9.81,
            'P_out_Pa':       self.P_out,
            'P_out_kPa':      self.P_out / 1_000.0,
            'T_out':          self.T_out,
            'rho_out':        rho_out,
            'fouling_factor': self.fouling_factor,
            'dp_ratio_pct':   self.delta_P / self.P_in * 100.0,
            'Q_Nm3h':         self.Q_gas_Nm3h,
        }
        self.state = result
        return result

    def step(self, dt: float, time: float) -> dict:
        t_hours = time / 3_600.0
        res = self.compute(t_hours)
        # Log simplifié
        self.history.append({'time': time, **res})
        if len(self.history) > 300:
            self.history.pop(0)
        return res

    def reset(self):
        self.delta_P        = self.DP_REF
        self.P_out          = self.P_in - self.delta_P
        self.T_out          = self.T_in
        self.fouling_factor = 1.0
        self.history        = []
        self.state          = {}