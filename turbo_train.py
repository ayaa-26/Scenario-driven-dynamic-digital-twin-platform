#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train turbosoufflante : Turbine vapeur + Soufflante centrifuge (arbre commun).
"""

from __future__ import annotations
import numpy as np
from steam_turbine import SteamTurbine
from turbo_blower   import TurboBlower


class TurboBlowerTrain:
    """Ensemble turbine vapeur + turbosoufflante (couplage par puissance)."""

    CONTROL_SPEED    = "speed"
    CONTROL_PRESSURE = "pressure"
    CONTROL_VALVE    = "valve"
    CONTROL_POWER    = "power"

    def __init__(self, name: str = "Turbosoufflante", tag: str = "401AC01/02"):
        self.name = name
        self.tag  = tag

        self.blower  = TurboBlower(name="Compresseurs principaux", tag=tag)
        self.turbine = SteamTurbine(name="Turbine vapeur entraînement",
                                    tag=f"{tag}-ST")

        self.eta_drive = 0.985

        self.control_mode       = self.CONTROL_SPEED
        self.target_delta_P_Pa  = None
        self.target_P_out_Pa    = None
        self.steam_valve        = 1.0
        self.available_power_kW = None

        self.state   = {}
        self.history = []

    # ------------------------------------------------------------------
    # Propriétés proxy vers blower
    # ------------------------------------------------------------------
    @property
    def load(self) -> float:
        return float(self.blower.load)

    @load.setter
    def load(self, value: float):
        self.blower.load = float(value)

    @property
    def T_in(self) -> float:
        return float(self.blower.T_in)

    @T_in.setter
    def T_in(self, value: float):
        self.blower.T_in = float(value)

    @property
    def P_in(self) -> float:
        return float(self.blower.P_in)

    @P_in.setter
    def P_in(self, value: float):
        self.blower.P_in = float(value)

    @property
    def Q_gas_Nm3h(self) -> float:
        return float(self.blower.Q_gas_Nm3h)

    @Q_gas_Nm3h.setter
    def Q_gas_Nm3h(self, value: float):
        self.blower.Q_gas_Nm3h = float(value)

    # ------------------------------------------------------------------
    # Modes de calcul internes
    # ------------------------------------------------------------------
    def _compute_speed_mode(self, load_sp: float):
        self.blower.load = float(load_sp)
        blower_res = self.blower.compute(self.blower.load)
        W_req_turb = blower_res["W_shaft_kW"] / max(self.eta_drive, 1e-9)
        valve_req  = self.turbine.valve_for_power(W_req_turb)
        power_limited = False

        if valve_req < 1.0 - 1e-9:
            self.steam_valve = float(valve_req)
            turbine_res = self.turbine.compute(self.steam_valve)
            return blower_res, turbine_res, self.blower.load, power_limited

        self.steam_valve = 1.0
        turbine_res = self.turbine.compute(self.steam_valve)
        W_drive = turbine_res["W_shaft_kW"] * self.eta_drive
        if W_drive + 1e-6 < blower_res["W_shaft_kW"]:
            power_limited = True
            load_act = self.blower.load_for_shaft_power(W_drive)
            self.blower.load = float(load_act)
            blower_res = self.blower.compute(self.blower.load)
        return blower_res, turbine_res, self.blower.load, power_limited

    def _compute_pressure_mode(self, delta_p_target: float):
        load_sp = self.blower.load_for_delta_p(delta_p_target)
        return self._compute_speed_mode(load_sp)

    def _compute_valve_mode(self, valve: float):
        self.steam_valve = float(np.clip(valve, 0.0, 1.0))
        turbine_res = self.turbine.compute(self.steam_valve)
        W_drive  = turbine_res["W_shaft_kW"] * self.eta_drive
        load_act = self.blower.load_for_shaft_power(W_drive)
        self.blower.load = float(load_act)
        blower_res = self.blower.compute(self.blower.load)
        return blower_res, turbine_res, self.blower.load, False

    def _compute_power_mode(self, W_turbine_kW: float):
        W_turb   = max(float(W_turbine_kW), 0.0)
        self.available_power_kW = W_turb
        W_drive  = W_turb * self.eta_drive
        load_act = self.blower.load_for_shaft_power(W_drive)
        self.blower.load = float(load_act)
        blower_res = self.blower.compute(self.blower.load)
        W_req_turb = blower_res["W_shaft_kW"] / max(self.eta_drive, 1e-9)
        self.steam_valve = float(self.turbine.valve_for_power(W_req_turb))
        turbine_res = self.turbine.compute(self.steam_valve)
        return blower_res, turbine_res, self.blower.load, False

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------
    def compute(self, load: float | None = None) -> dict:
        if load is not None:
            self.control_mode = self.CONTROL_SPEED
            self.blower.load  = float(load)

        mode = self.control_mode

        if mode == self.CONTROL_PRESSURE:
            if self.target_P_out_Pa is not None:
                delta_p = float(self.target_P_out_Pa) - float(self.blower.P_in)
            else:
                delta_p = float(self.target_delta_P_Pa
                                if self.target_delta_P_Pa is not None
                                else self.blower.DP_NOM)
            blower_res, turbine_res, load_act, power_limited = \
                self._compute_pressure_mode(delta_p)

        elif mode == self.CONTROL_VALVE:
            blower_res, turbine_res, load_act, power_limited = \
                self._compute_valve_mode(self.steam_valve)

        elif mode == self.CONTROL_POWER:
            blower_res, turbine_res, load_act, power_limited = \
                self._compute_power_mode(
                    float(self.available_power_kW
                          if self.available_power_kW is not None else 0.0))
        else:  # CONTROL_SPEED (défaut)
            blower_res, turbine_res, load_act, power_limited = \
                self._compute_speed_mode(self.blower.load)

        W_turb = float(turbine_res["W_shaft_kW"])
        W_drive = W_turb * self.eta_drive
        W_blow  = float(blower_res["W_shaft_kW"])

        result = {
            **blower_res,
            "control_mode"     : mode,
            "load_actual"      : float(load_act),
            "steam_valve_pct"  : float(self.steam_valve) * 100.0,
            "steam_mdot_kg_s"  : float(turbine_res["mdot_kg_s"]),
            "W_turbine_kW"     : W_turb,
            "W_drive_kW"       : W_drive,
            "power_margin_kW"  : W_drive - W_blow,
            "power_limited"    : bool(power_limited),
            "eta_drive"        : float(self.eta_drive),
            "turbine_T_out_C"  : float(turbine_res["T_out_C"]),
            "turbine_P_out_Pa" : float(turbine_res["P_out_Pa"]),
        }
        self.state = result
        return result

    def step(self, dt: float, time: float) -> dict:
        res = self.compute()
        self.history.append({"time": time, **res})
        return res

    def reset(self):
        self.blower.reset()
        self.turbine.reset()
        self.control_mode       = self.CONTROL_SPEED
        self.target_delta_P_Pa  = None
        self.target_P_out_Pa    = None
        self.steam_valve        = 1.0
        self.available_power_kW = None
        self.history            = []
        self.state              = {}