"""
bac.py
======
Modèle dynamique des bacs de stockage — usine acide sulfurique OCP Jorf Lasfar.

Basé sur les équations 153-156 de la thèse :

  Eq.153  A · dh_B/dt       = ΣF_e,i  − ΣF_s,j                         (bilan volumique)
  Eq.154  A·h_B · dd_s/dt   = Σ d_e,i·F_e,i  − d_s · ΣF_e,i            (bilan massique)
  Eq.155  A·h_B · dC_i/dt   = Σ C^i_e,j·F_e,j − Σ C^i_e,k·F_s,k
          + C_i·(ΣF_s,j − ΣF_e,i)                                        (bilan composition)
  Eq.156  d_s·A·h_B·cp·dT/dt = Σ d_e,i·cp_e,i·F_e,i·(T_e,i−T_ref)
          − (4·U_loss/D_r)·(T−T_s) − (T−T_ref)·cp·Σ d_e,i·F_e,i        (bilan thermique)

Trois sous-classes préconfigurées :
  - SulfurTank  : bac soufre liquide jaune  (liquide, ~135-145°C, ρ~1760 kg/m³)
  - AcidTank    : bac H₂SO₄ vert           (liquide, ~30-70°C,   ρ~1830 kg/m³)
  - GenericTank : bac générique paramétrable
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE FLUX
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FlowIn:
    """
    Flux volumique entrant dans un bac.

    Paramètres
    ----------
    name       : identifiant du flux
    F          : débit volumique [m³/s]
    density    : densité du fluide entrant [kg/m³]
    T          : température du fluide entrant [°C]
    cp         : capacité calorifique massique [J/(kg·K)]
    composition: dict {composé: fraction massique ou concentration [kg/m³]}
    """
    name:        str
    F:           float = 0.0          # m³/s
    density:     float = 1000.0       # kg/m³
    T:           float = 25.0         # °C
    cp:          float = 4180.0       # J/(kg·K)
    composition: Dict[str, float] = field(default_factory=dict)


@dataclass
class FlowOut:
    """
    Flux volumique sortant d'un bac.

    Paramètres
    ----------
    name : identifiant du flux
    F    : débit volumique [m³/s]
    """
    name: str
    F:    float = 0.0   # m³/s


# ══════════════════════════════════════════════════════════════════════
# MODÈLE DYNAMIQUE GÉNÉRIQUE — ÉQUATIONS 153-156
# ══════════════════════════════════════════════════════════════════════

class GenericTank:
    """
    Modèle dynamique d'un bac de stockage basé sur les équations 153-156.

    État interne
    ------------
    h_B   : niveau [m]
    d_s   : densité du contenu [kg/m³]
    T     : température [°C]
    C     : dict {composé: concentration [kg/m³]}

    Paramètres géométriques
    -----------------------
    A     : section transversale [m²]
    h_max : niveau maximum [m]
    h_min : niveau minimum (fond mort) [m]
    D_r   : diamètre équivalent [m]  (pour pertes thermiques)

    Paramètres thermiques
    ---------------------
    cp        : capacité calorifique massique contenu [J/(kg·K)]
    U_loss    : coefficient global pertes thermiques [W/(m²·K)]
    T_s       : température ambiante / enveloppe [°C]
    T_ref     : température de référence (enthalpie nulle) [°C]
    """

    def __init__(
        self,
        name:      str,
        tag:       str,
        # Géométrie
        A:         float = 100.0,   # m² (section)
        h_max:     float = 10.0,    # m
        h_min:     float = 0.20,    # m
        D_r:       float = 11.3,    # m (diamètre équivalent ≈ √(4A/π))
        # État initial
        h_init:    float = 5.0,     # m
        d_init:    float = 1000.0,  # kg/m³
        T_init:    float = 25.0,    # °C
        # Paramètres thermiques
        cp:        float = 4180.0,  # J/(kg·K)
        U_loss:    float = 0.5,     # W/(m²·K)
        T_s:       float = 30.0,    # °C  (ambiance)
        T_ref:     float = 25.0,    # °C  (référence enthalpie)
        # Composition initiale
        composition_init: Dict[str, float] | None = None,
    ):
        self.name  = name
        self.tag   = tag

        # Géométrie
        self.A     = A
        self.h_max = h_max
        self.h_min = h_min
        self.D_r   = D_r if D_r > 0 else max(math.sqrt(4 * A / math.pi), 0.1)

        # Paramètres thermiques
        self.cp    = cp
        self.U_loss = U_loss
        self.T_s   = T_s
        self.T_ref = T_ref

        # État dynamique
        self.h_B = float(h_init)
        self.d_s = float(d_init)
        self.T   = float(T_init)
        self.C: Dict[str, float] = dict(composition_init) if composition_init else {}

        # Historique pour Streamlit / tracés
        self.history: list[dict] = []
        self.time:    float = 0.0

        # Flux courants (mis à jour avant chaque step)
        self.inflows:  List[FlowIn]  = []
        self.outflows: List[FlowOut] = []

        # Résultats calculés
        self.state: dict = {}
        self._log_state()

    # ── Propriétés dérivées ───────────────────────────────────────────

    @property
    def volume(self) -> float:
        """Volume de liquide [m³]."""
        return self.A * max(self.h_B, 0.0)

    @property
    def mass(self) -> float:
        """Masse de liquide [kg]."""
        return self.d_s * self.volume

    @property
    def level_pct(self) -> float:
        """Niveau en % du h_max."""
        return min(max(self.h_B / self.h_max * 100.0, 0.0), 100.0)

    @property
    def sum_Fe(self) -> float:
        """Somme des débits entrants [m³/s]."""
        return sum(f.F for f in self.inflows)

    @property
    def sum_Fs(self) -> float:
        """Somme des débits sortants [m³/s]."""
        return sum(f.F for f in self.outflows)

    # ── Calcul des dérivées (équations 153-156) ───────────────────────

    def _derivatives(self) -> tuple[float, float, float, Dict[str, float]]:
        """
        Calcule les dérivées selon les équations du modèle.

        Retour : (dh_dt, dd_dt, dT_dt, dC_dt_dict)
        """
        A    = self.A
        h_B  = max(self.h_B, 1e-6)
        d_s  = self.d_s
        T    = self.T
        cp   = self.cp

        sum_Fe = self.sum_Fe
        sum_Fs = self.sum_Fs

        # ── Eq.153 : bilan volumique ──────────────────────────────────
        # A · dh/dt = ΣF_e,i − ΣF_s,j
        dh_dt = (sum_Fe - sum_Fs) / A

        # ── Eq.154 : bilan densité ────────────────────────────────────
        # A·h_B · dd_s/dt = Σ(d_e,i · F_e,i) − d_s · ΣF_e,i
        sum_de_Fe = sum(f.density * f.F for f in self.inflows)
        if h_B > 1e-6:
            dd_dt = (sum_de_Fe - d_s * sum_Fe) / (A * h_B)
        else:
            dd_dt = 0.0

        # ── Eq.156 : bilan thermique ──────────────────────────────────
        # d_s·A·h_B·cp · dT/dt
        #   = Σ(d_e,i·cp_e,i·F_e,i·(T_e,i−T_ref))
        #     − (4·U_loss/D_r)·(T−T_s)
        #     − (T−T_ref)·cp·Σ(d_e,i·F_e,i)
        V = A * h_B
        inflow_enthalpy = sum(
            f.density * f.cp * f.F * (f.T - self.T_ref)
            for f in self.inflows
        )
        loss_term = (4.0 * self.U_loss / self.D_r) * (T - self.T_s)
        outflow_term = (T - self.T_ref) * cp * sum_de_Fe

        denom_T = d_s * V * cp
        if denom_T > 1e-6:
            dT_dt = (inflow_enthalpy - loss_term - outflow_term) / denom_T
        else:
            dT_dt = 0.0

        # ── Eq.155 : bilan composition ────────────────────────────────
        # A·h_B · dC_i/dt = Σ(C^i_e,j · F_e,j) − Σ(C^i_e,k · F_s,k)
        #                   + C_i·(ΣF_s,j − ΣF_e,i)
        # Les flux sortants ont la composition du bac (hypothèse mélange parfait)
        dC_dt: Dict[str, float] = {}
        if h_B > 1e-6:
            all_species = set(self.C.keys())
            for f in self.inflows:
                all_species.update(f.composition.keys())
            for species in all_species:
                C_i    = self.C.get(species, 0.0)
                in_flux = sum(
                    f.composition.get(species, 0.0) * f.F
                    for f in self.inflows
                )
                out_flux = C_i * sum_Fs   # hypothèse mélange parfait
                net_vol  = C_i * (sum_Fs - sum_Fe)
                dC_dt[species] = (in_flux - out_flux + net_vol) / (A * h_B)
        return dh_dt, dd_dt, dT_dt, dC_dt

    # ── Intégration explicite (Euler) ─────────────────────────────────

    def step(self, dt: float) -> dict:
        """
        Avance l'état du bac d'un pas de temps dt [s].
        Intégration Euler explicite avec limitation physique.
        """
        dh, dd, dT, dC = self._derivatives()

        # Mise à jour niveau
        self.h_B = max(self.h_min, min(self.h_max, self.h_B + dh * dt))

        # Mise à jour densité (bornée physiquement)
        self.d_s = max(500.0, min(2500.0, self.d_s + dd * dt))

        # Mise à jour température (bornée physiquement)
        self.T = max(-10.0, min(400.0, self.T + dT * dt))

        # Mise à jour composition
        for sp, dCdt in dC.items():
            self.C[sp] = max(0.0, self.C.get(sp, 0.0) + dCdt * dt)

        self.time += dt
        self._log_state()
        return self.state

    def compute(self, dt: float | None = None) -> dict:
        """Interface compatible avec les équipements existants."""
        if dt is not None:
            return self.step(dt)
        self._log_state()
        return self.state

    # ── Logging ───────────────────────────────────────────────────────

    def _log_state(self):
        self.state = {
            "name":       self.name,
            "tag":        self.tag,
            "time_s":     round(self.time, 2),
            "h_B_m":      round(self.h_B, 4),
            "level_pct":  round(self.level_pct, 2),
            "volume_m3":  round(self.volume, 2),
            "mass_kg":    round(self.mass, 1),
            "density_kg_m3": round(self.d_s, 2),
            "T_C":        round(self.T, 2),
            "sum_Fe_m3s": round(self.sum_Fe, 5),
            "sum_Fs_m3s": round(self.sum_Fs, 5),
        }
        for sp, val in self.C.items():
            self.state[f"C_{sp}_kg_m3"] = round(val, 4)
        self.history.append(dict(self.state))

    def reset(self, h_init=None, T_init=None, d_init=None):
        """Remet le bac à son état initial (ou valeurs fournies)."""
        if h_init is not None:
            self.h_B = float(h_init)
        if T_init is not None:
            self.T = float(T_init)
        if d_init is not None:
            self.d_s = float(d_init)
        self.time = 0.0
        self.history = []
        self._log_state()

    # ── Utilitaires ───────────────────────────────────────────────────

    def set_inflows(self, inflows: List[FlowIn]):
        self.inflows = inflows

    def set_outflows(self, outflows: List[FlowOut]):
        self.outflows = outflows

    def add_inflow(self, flow: FlowIn):
        self.inflows.append(flow)

    def add_outflow(self, flow: FlowOut):
        self.outflows.append(flow)

    def __repr__(self) -> str:
        return (
            f"{self.name} [{self.tag}]  "
            f"h={self.h_B:.2f}/{self.h_max:.1f} m  "
            f"({self.level_pct:.1f}%)  "
            f"T={self.T:.1f}°C  ρ={self.d_s:.0f} kg/m³"
        )


# ══════════════════════════════════════════════════════════════════════
# BAC SOUFRE LIQUIDE — JAUNE
# ══════════════════════════════════════════════════════════════════════

class SulfurTank(GenericTank):
    """
    Bac de stockage de soufre liquide (couleur jaune dans le P&ID).

    Caractéristiques physiques du soufre liquide :
      - Densité         : 1760 kg/m³  (à 135°C)
      - T de fusion     : 119°C
      - T opératoire    : 135-145°C  (maintenu fondu par traçage vapeur)
      - cp              : 1010 J/(kg·K)
      - Composition     : {S: 1.0}  (pureté ≈ 99.9 %)

    Deux bacs identiques en parallèle sur le site OCP Jorf Lasfar.
    Chaque bac reçoit du soufre liquide depuis les camions/wagons
    et alimente le four à soufre (401-AF-01) via une pompe.
    """

    # Paramètres par défaut OCP (estimés d'après schéma)
    DEFAULT_A       = 380.0    # m²  (bac cylindrique D≈22 m)
    DEFAULT_H_MAX   = 8.0     # m
    DEFAULT_H_MIN   = 0.30    # m
    DEFAULT_D_R     = 22.0    # m
    DEFAULT_H_INIT  = 5.0     # m  (~63% niveau nominal)
    DEFAULT_D_INIT  = 1760.0  # kg/m³
    DEFAULT_T_INIT  = 138.0   # °C
    DEFAULT_CP      = 1010.0  # J/(kg·K)
    DEFAULT_U_LOSS  = 1.2     # W/(m²·K)  (bac isolé, traçage vapeur)
    DEFAULT_T_S     = 35.0    # °C  (ambiance extérieure)
    DEFAULT_T_REF   = 25.0    # °C

    def __init__(
        self,
        name: str = "Bac soufre",
        tag:  str = "BAC-S",
        **kwargs,
    ):
        params = dict(
            A             = self.DEFAULT_A,
            h_max         = self.DEFAULT_H_MAX,
            h_min         = self.DEFAULT_H_MIN,
            D_r           = self.DEFAULT_D_R,
            h_init        = self.DEFAULT_H_INIT,
            d_init        = self.DEFAULT_D_INIT,
            T_init        = self.DEFAULT_T_INIT,
            cp            = self.DEFAULT_CP,
            U_loss        = self.DEFAULT_U_LOSS,
            T_s           = self.DEFAULT_T_S,
            T_ref         = self.DEFAULT_T_REF,
            composition_init = {"S": 1760.0 * 0.999},  # ~99.9% S en kg/m³
        )
        params.update(kwargs)
        # Attributs spécifiques soufre (avant super() qui appelle _log_state)
        self.T_solidification = 119.0
        self.T_max_safe       = 160.0
        self.purity_pct       = 99.9
        super().__init__(name=name, tag=tag, **params)

        # Paramètres spécifiques soufre
        self.T_solidification = 119.0   # °C — seuil d'alarme solidification
        self.T_max_safe       = 160.0   # °C — seuil d'alarme surchauffe
        self.purity_pct       = 99.9    # %

    @property
    def is_solidification_risk(self) -> bool:
        """Alerte si température trop basse (risque solidification soufre)."""
        return self.T < self.T_solidification + 5.0

    @property
    def is_overheat_risk(self) -> bool:
        """Alerte si température trop élevée."""
        return self.T > self.T_max_safe

    @property
    def flow_out_kg_per_min(self) -> float:
        """Débit sortant en kg/min (pour affichage four à soufre)."""
        return self.d_s * self.sum_Fs * 60.0

    def _log_state(self):
        super()._log_state()
        self.state["solidification_risk"] = self.is_solidification_risk
        self.state["overheat_risk"]        = self.is_overheat_risk
        self.state["flow_out_kg_min"]      = round(self.flow_out_kg_per_min, 2)
        self.state["purity_pct"]           = round(self.purity_pct, 2)


# ══════════════════════════════════════════════════════════════════════
# BAC ACIDE SULFURIQUE — VERT
# ══════════════════════════════════════════════════════════════════════

class AcidTank(GenericTank):
    """
    Bac de stockage d'acide sulfurique produit (couleur verte dans le P&ID).

    Caractéristiques physiques de H₂SO₄ concentré (98%) :
      - Densité         : 1830 kg/m³  (à 25°C, 98% H₂SO₄)
      - T opératoire    : 30-70°C  (acide concentré en sortie des tours)
      - cp              : 1420 J/(kg·K)
      - Composition     : {H2SO4: fraction massique ≈ 0.98}

    Reçoit l'acide depuis les tours d'absorption (JD02/JD03)
    et alimente les camions-citernes / expédition.
    """

    DEFAULT_A       = 530.0    # m²  (bac cylindrique D≈26 m)
    DEFAULT_H_MAX   = 12.0    # m
    DEFAULT_H_MIN   = 0.50    # m
    DEFAULT_D_R     = 26.0    # m
    DEFAULT_H_INIT  = 7.0     # m
    DEFAULT_D_INIT  = 1830.0  # kg/m³  (H₂SO₄ 98%)
    DEFAULT_T_INIT  = 45.0    # °C
    DEFAULT_CP      = 1420.0  # J/(kg·K)
    DEFAULT_U_LOSS  = 0.4     # W/(m²·K)  (bac acide, bon isolant)
    DEFAULT_T_S     = 35.0    # °C
    DEFAULT_T_REF   = 25.0    # °C

    # Concentration massique H₂SO₄ à 98% et ρ=1830 kg/m³
    _C_H2SO4_INIT   = 1830.0 * 0.98   # kg/m³ ≈ 1793.4

    def __init__(
        self,
        name: str = "Bac acide sulfurique",
        tag:  str = "BAC-A",
        **kwargs,
    ):
        params = dict(
            A             = self.DEFAULT_A,
            h_max         = self.DEFAULT_H_MAX,
            h_min         = self.DEFAULT_H_MIN,
            D_r           = self.DEFAULT_D_R,
            h_init        = self.DEFAULT_H_INIT,
            d_init        = self.DEFAULT_D_INIT,
            T_init        = self.DEFAULT_T_INIT,
            cp            = self.DEFAULT_CP,
            U_loss        = self.DEFAULT_U_LOSS,
            T_s           = self.DEFAULT_T_S,
            T_ref         = self.DEFAULT_T_REF,
            composition_init = {
                "H2SO4": self._C_H2SO4_INIT,
                "H2O":   1830.0 * 0.02,
            },
        )
        params.update(kwargs)
        # Attributs spécifiques acide (avant super() qui appelle _log_state)
        self.T_max_safe    = 80.0
        self.w_H2SO4_min   = 94.0
        self.w_H2SO4_max   = 99.5
        super().__init__(name=name, tag=tag, **params)

        # Paramètres spécifiques acide
        self.T_max_safe    = 80.0   # °C — au-delà risque vapeurs / corrosion accélérée
        self.w_H2SO4_min   = 94.0   # %  — titre minimum acceptable
        self.w_H2SO4_max   = 99.5   # %  — titre maximum (oleum si >99%)

    @property
    def w_H2SO4_pct(self) -> float:
        """Titre massique H₂SO₄ [%]."""
        C_acid = self.C.get("H2SO4", 0.0)
        if self.d_s > 0:
            return min(max(C_acid / self.d_s * 100.0, 0.0), 100.0)
        return 0.0

    @property
    def is_overheat_risk(self) -> bool:
        return self.T > self.T_max_safe

    @property
    def is_dilution_risk(self) -> bool:
        """Alerte si titre trop faible."""
        return self.w_H2SO4_pct < self.w_H2SO4_min

    def _log_state(self):
        super()._log_state()
        self.state["w_H2SO4_pct"]    = round(self.w_H2SO4_pct, 2)
        self.state["overheat_risk"]   = self.is_overheat_risk
        self.state["dilution_risk"]   = self.is_dilution_risk


# ══════════════════════════════════════════════════════════════════════
# SYSTÈME GLOBAL : 2 BACS SOUFRE + 1 BAC ACIDE
# ══════════════════════════════════════════════════════════════════════

class TankSystem:
    """
    Système des 3 bacs interconnectés :
      - sulfur_tank_1 : Bac soufre n°1 (jaune)
      - sulfur_tank_2 : Bac soufre n°2 (jaune)
      - acid_tank     : Bac acide sulfurique (vert)

    Les débits (F_e, F_s) sont mis à jour par l'utilisateur à chaque pas de temps
    avant d'appeler step(). Le débit sortant des bacs soufre correspond
    au débit entrant dans le four (S_kgm = kg/min → converti en m³/s).
    """

    def __init__(self):
        self.sulfur_tank_1 = SulfurTank(
            name="Bac soufre 1",
            tag="BAC-S-01",
        )
        self.sulfur_tank_2 = SulfurTank(
            name="Bac soufre 2",
            tag="BAC-S-02",
            h_init=4.0,   # niveau légèrement différent pour différencier
        )
        self.acid_tank = AcidTank(
            name="Bac acide sulfurique",
            tag="BAC-A-01",
        )

        # Paramètres de flux nominaux
        self._S_kgmin_nom  = 800.0   # kg/min — débit soufre nominal au four
        self._acid_prod_m3s = 0.0    # m³/s — production acide (mis à jour depuis simulation)

        self.time = 0.0

    # ── Mise à jour des flux depuis la simulation principale ──────────

    def update_flows(
        self,
        S_kgmin:         float = 800.0,   # kg/min sortant des bacs soufre → four
        acid_prod_m3s:   float = 0.0,     # m³/s entrant dans le bac acide (depuis tours abs.)
        acid_out_m3s:    float = 0.0,     # m³/s sortant du bac acide → expédition
        sulfur_in_1_m3s: float = 0.0,     # m³/s entrant bac soufre 1 (livraisons)
        sulfur_in_2_m3s: float = 0.0,     # m³/s entrant bac soufre 2
        split_ratio:     float = 0.5,     # fraction du débit four depuis bac 1 [0-1]
        acid_T_in:       float = 55.0,    # °C — T acide entrant depuis tours
        sulfur_T_in:     float = 140.0,   # °C — T soufre livré
    ):
        """Configure les flux de tous les bacs en cohérence avec la simulation."""
        # Débit soufre → four : converti kg/min → m³/s
        S_m3s = (S_kgmin / 60.0) / self.sulfur_tank_1.d_s

        # Bac soufre 1
        self.sulfur_tank_1.set_inflows([
            FlowIn(
                name="Livraison soufre",
                F=sulfur_in_1_m3s,
                density=1760.0,
                T=sulfur_T_in,
                cp=1010.0,
                composition={"S": 1760.0 * 0.999},
            )
        ])
        self.sulfur_tank_1.set_outflows([
            FlowOut(name="Vers four 401-AF-01", F=S_m3s * split_ratio)
        ])

        # Bac soufre 2
        self.sulfur_tank_2.set_inflows([
            FlowIn(
                name="Livraison soufre",
                F=sulfur_in_2_m3s,
                density=1760.0,
                T=sulfur_T_in,
                cp=1010.0,
                composition={"S": 1760.0 * 0.999},
            )
        ])
        self.sulfur_tank_2.set_outflows([
            FlowOut(name="Vers four 401-AF-01", F=S_m3s * (1.0 - split_ratio))
        ])

        # Bac acide
        rho_acid = self.acid_tank.d_s
        w_acid   = self.acid_tank.w_H2SO4_pct / 100.0
        self.acid_tank.set_inflows([
            FlowIn(
                name="Production tours absorption",
                F=acid_prod_m3s,
                density=rho_acid,
                T=acid_T_in,
                cp=1420.0,
                composition={
                    "H2SO4": rho_acid * w_acid,
                    "H2O":   rho_acid * (1.0 - w_acid),
                },
            )
        ])
        self.acid_tank.set_outflows([
            FlowOut(name="Expédition / chargement camions", F=acid_out_m3s)
        ])

    def step(self, dt: float) -> dict:
        """Avance tous les bacs d'un pas de temps dt [s]."""
        r1 = self.sulfur_tank_1.step(dt)
        r2 = self.sulfur_tank_2.step(dt)
        ra = self.acid_tank.step(dt)
        self.time += dt
        return {
            "sulfur_tank_1": r1,
            "sulfur_tank_2": r2,
            "acid_tank":     ra,
            "time_s":        self.time,
        }

    def get_state(self) -> dict:
        return {
            "sulfur_tank_1": self.sulfur_tank_1.state,
            "sulfur_tank_2": self.sulfur_tank_2.state,
            "acid_tank":     self.acid_tank.state,
        }

    # ── Indicateurs globaux ───────────────────────────────────────────

    @property
    def total_sulfur_mass_t(self) -> float:
        """Masse totale soufre stockée [tonnes]."""
        return (self.sulfur_tank_1.mass + self.sulfur_tank_2.mass) / 1000.0

    @property
    def total_acid_mass_t(self) -> float:
        """Masse totale acide stockée [tonnes]."""
        return self.acid_tank.mass / 1000.0

    @property
    def autonomy_hours_sulfur(self) -> float:
        """
        Autonomie en heures avec le stock soufre actuel,
        au débit courant vers le four.
        """
        total_F_out = (self.sulfur_tank_1.sum_Fs
                       + self.sulfur_tank_2.sum_Fs)   # m³/s
        rho = (self.sulfur_tank_1.d_s + self.sulfur_tank_2.d_s) / 2.0
        if total_F_out * rho > 1e-6:
            # heures = masse_totale_kg / (débit_massique_total_kg_s × 3600)
            debit_kg_s = total_F_out * rho
            return self.total_sulfur_mass_t * 1000.0 / debit_kg_s / 3600.0
        return float('inf')