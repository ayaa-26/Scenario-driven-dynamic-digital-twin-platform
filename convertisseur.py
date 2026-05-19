# =====================================================================
# model/convertisseur.py
# Jumeau numérique DYNAMIQUE du convertisseur catalytique (4 lits)
#
# Passage du modèle statique (profils ODE en z) vers un modèle dynamique
# discrétisé en volumes finis (z, t) avec :
#   - Bilan matière transitoire : ∂τ/∂t + u·∂τ/∂z = S_τ(τ, T)
#   - Bilan énergie transitoire : ∂T/∂t + u·∂T/∂z = S_T(τ, T)
# Schéma upwind explicite + RK4 en temps.
# =====================================================================

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Structures de données
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ConditionsEntree:
    """Conditions opératoires à l'entrée du convertisseur."""
    Q_in_m3h: float = 624_600.0          # débit volumique [m³/h]
    P_in_Pa: float = 150_000.0           # pression [Pa]
    T_entree_C: float = 420.0            # température entrée lit 1 [°C]
    y0: Dict[str, float] = field(default_factory=lambda: {
        'SO2': 0.1101,
        'O2':  0.0999,
        'SO3': 0.000165,
        'N2':  1.0 - (0.1101 + 0.0999 + 0.000165),
    })
    T_entree_lits_C: List[float] = field(default_factory=lambda: [420.0, 454.0, 449.0, 425.0])


@dataclass
class EtatLit:
    """État dynamique (τ, T) discrétisé sur N_z cellules d'un lit."""
    tau: np.ndarray     # taux de conversion local [-]
    T_K: np.ndarray     # température locale [K]

    def copy(self) -> "EtatLit":
        return EtatLit(self.tau.copy(), self.T_K.copy())


# ─────────────────────────────────────────────────────────────────────
# Jumeau numérique dynamique
# ─────────────────────────────────────────────────────────────────────

class JumeauNumeriqueConvertisseur:
    """
    Modèle dynamique 1-D (z, t) d'un convertisseur catalytique à 4 lits.

    Équations gouvernantes (forme conservative, upwind explicite) :
        ∂τ/∂t  =  −u·(τ_i − τ_{i-1})/Δz  +  S_τ(τ_i, T_i)
        ∂T/∂t  =  −u·(T_i − T_{i-1})/Δz  +  S_T(τ_i, T_i)

    où :
        u   = vitesse interstitielle du gaz [m/s]
        S_τ = terme source conversion  = (1−ε)·ρ_cat·A·r / F_SO2_in
        S_T = terme source énergie     = (1−ε)·ρ_cat·A·(−ΔH_r)·r / (Q_in·ρ_g·cp_g)

    Stabilité CFL : dt ≤ dz / u   (vérifié automatiquement).
    """

    # ------------------------------------------------------------------
    # Constantes physiques & paramètres du procédé
    # ------------------------------------------------------------------
    R        = 8.314        # constante des gaz parfaits [J·mol⁻¹·K⁻¹]
    deltaH_r = -99_000.0    # enthalpie de réaction SO2→SO3 [J/mol]
    rho_cat  = 550.0        # masse volumique du lit catalytique [kg/m³]
    epsilon  = 0.523        # porosité du lit [-]
    Dr       = 12.8         # diamètre du réacteur [m]   (non utilisé directement)
    A_lit    = 128.68       # section transversale [m²]
    D_p      = 0.005        # diamètre des particules [m]

    V_cat_lits = [87.0, 103.0, 134.0, 167.0]   # volumes de catalyseur [m³]
    facteur_activite_lits = [4500.0, 11_000.0, 18_000.0, 45_000.0]

    SHOMATE: Dict[str, List[float]] = {
        'N2':  [19.50583,  19.88705,  -8.598535,  1.369784,  0.527601],
        'O2':  [30.03235,   8.772972, -3.988133,  0.788313, -0.741599],
        'SO2': [21.43049,  74.35094, -57.75217,  16.35534,  0.086731],
        'SO3': [24.02503, 119.4607,  -94.38686,  26.96237, -0.117517],
    }
    MOLAR_MASS: Dict[str, float] = {
        'SO2': 0.06407, 'O2': 0.03200,
        'N2':  0.02801, 'SO3': 0.08006,
    }
    cinetique = {
        'K1': {'A': 2.15e13, 'E': 98_900.0},
        'K3': {'A': 7.8e-2,  'E':  6_280.0},
        'K4': {'A': 1.3e4,   'E': 25_100.0},
    }
    mu_ref: Dict[str, float] = {
        'SO2': 1.23e-5, 'O2': 1.92e-5, 'N2': 1.66e-5, 'SO3': 1.45e-5,
    }

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, conditions: Optional[ConditionsEntree] = None, N_z: int = 100):
        """
        Parameters
        ----------
        conditions : ConditionsEntree, optionnel
            Conditions opératoires initiales.
        N_z : int
            Nombre de cellules de discrétisation spatiale par lit.
        """
        self.cond  = conditions or ConditionsEntree()
        self.N_z   = N_z

        # Dérivés géométriques
        self.H_cat_reel = [V / self.A_lit for V in self.V_cat_lits]
        self.n_lits     = len(self.H_cat_reel)

        # Débit volumique en m³/s
        self.Q_in = self.cond.Q_in_m3h / 3600.0

        # Vitesse superficielle initiale (m/s) – recalculée à chaque pas de temps
        self._u_superficielle: float = self.Q_in / self.A_lit

        # ── État dynamique ──────────────────────────────────────────
        # Chaque lit : tableau τ[N_z], T[N_z] discrétisé en z
        self.etats: List[EtatLit] = self._initialiser_etats()

        # Historique pour le post-traitement
        self.historique_temps: List[float] = []
        self.historique_tau_sortie: List[float] = []   # τ cumulé global [%]
        self.historique_T_sortie_lits: List[List[float]] = []
        self.t_simule: float = 0.0

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def set_conditions(
        self,
        Q_in_m3h: Optional[float] = None,
        P_in_Pa: Optional[float] = None,
        T_entree_C: Optional[float] = None,
        y0: Optional[Dict[str, float]] = None,
    ) -> None:
        """Met à jour les conditions opératoires à la volée (perturbation)."""
        if Q_in_m3h is not None:
            self.cond.Q_in_m3h = Q_in_m3h
            self.Q_in = Q_in_m3h / 3600.0
        if P_in_Pa is not None:
            self.cond.P_in_Pa = P_in_Pa
        if T_entree_C is not None:
            self.cond.T_entree_lits_C[0] = T_entree_C
        if y0 is not None:
            self.cond.y0 = y0

    def avancer(self, dt: float, n_steps: int = 1) -> None:
        """
        Intègre l'état dynamique sur n_steps pas de temps dt [s].

        Parameters
        ----------
        dt      : pas de temps [s] – doit satisfaire la condition CFL.
        n_steps : nombre de pas à effectuer.
        """
        for _ in range(n_steps):
            self._verifier_cfl(dt)
            self._pas_rk4(dt)
            self.t_simule += dt
            self._enregistrer_historique()

    def simuler_transitoire(
        self,
        duree_s: float,
        dt: float = 1.0,
        perturbations: Optional[List[Tuple[float, Dict]]] = None,
    ) -> Dict:
        """
        Lance une simulation dynamique complète sur `duree_s` secondes.

        Parameters
        ----------
        duree_s       : durée totale [s]
        dt            : pas de temps [s]
        perturbations : liste de (t_perturb, kwargs_set_conditions)
                        ex. [(300, {'T_entree_C': 430})]

        Returns
        -------
        dict contenant les séries temporelles et les profils finaux.
        """
        perturbations = sorted(perturbations or [], key=lambda x: x[0])
        idx_pert      = 0
        n_steps_total = int(np.ceil(duree_s / dt))

        for step in range(n_steps_total):
            t_courant = self.t_simule

            # Appliquer les perturbations au bon moment
            while idx_pert < len(perturbations) and perturbations[idx_pert][0] <= t_courant:
                self.set_conditions(**perturbations[idx_pert][1])
                idx_pert += 1

            self.avancer(dt)

        return self._construire_resultats_finaux()

    def etat_stationnaire(self, tol: float = 1e-4, dt: float = 1.0, max_iter: int = 10_000) -> Dict:
        """
        Intègre jusqu'au régime permanent (variation relative < tol).

        Returns
        -------
        dict des résultats au régime permanent.
        """
        tau_prev = np.inf
        for i in range(max_iter):
            self.avancer(dt)
            tau_curr = self.historique_tau_sortie[-1] if self.historique_tau_sortie else 0.0
            if i > 0 and abs(tau_curr - tau_prev) / (abs(tau_prev) + 1e-12) < tol:
                break
            tau_prev = tau_curr
        return self._construire_resultats_finaux()

    # ------------------------------------------------------------------
    # Initialisation des états (profil plat = conditions d'entrée)
    # ------------------------------------------------------------------

    def _initialiser_etats(self) -> List[EtatLit]:
        etats = []
        for i in range(self.n_lits):
            T0 = self.cond.T_entree_lits_C[i] + 273.15
            etats.append(EtatLit(
                tau=np.zeros(self.N_z),
                T_K=np.full(self.N_z, T0),
            ))
        return etats

    # ------------------------------------------------------------------
    # Intégration temporelle RK4
    # ------------------------------------------------------------------

    def _pas_rk4(self, dt: float) -> None:
        """Un pas RK4 sur tous les lits en cascade."""
        y_in = self.cond.y0.copy()
        tau_cumule_global = 0.0

        for i in range(self.n_lits):
            etat = self.etats[i]
            T_bc = self.cond.T_entree_lits_C[i] + 273.15

            # --- Conditions limites amont ---
            tau_in = 0.0
            T_in   = T_bc

            # RK4 sur le vecteur d'état [tau, T_K] du lit i
            k1_tau, k1_T = self._residus(etat.tau, etat.T_K, tau_in, T_in, y_in, i)
            tau2 = np.clip(etat.tau + 0.5*dt*k1_tau, 0.0, 0.999)
            T2   = etat.T_K + 0.5*dt*k1_T

            k2_tau, k2_T = self._residus(tau2, T2, tau_in, T_in, y_in, i)
            tau3 = np.clip(etat.tau + 0.5*dt*k2_tau, 0.0, 0.999)
            T3   = etat.T_K + 0.5*dt*k2_T

            k3_tau, k3_T = self._residus(tau3, T3, tau_in, T_in, y_in, i)
            tau4 = np.clip(etat.tau + dt*k3_tau, 0.0, 0.999)
            T4   = etat.T_K + dt*k3_T

            k4_tau, k4_T = self._residus(tau4, T4, tau_in, T_in, y_in, i)

            new_tau = np.clip(
                etat.tau + (dt/6.0)*(k1_tau + 2*k2_tau + 2*k3_tau + k4_tau),
                0.0, 0.999
            )
            new_T = etat.T_K + (dt/6.0)*(k1_T + 2*k2_T + 2*k3_T + k4_T)

            self.etats[i] = EtatLit(new_tau, new_T)

            # Mise à jour de la composition de sortie pour le lit suivant
            tau_sortie_local = new_tau[-1]
            y_in = self._composition_locale(y_in, tau_sortie_local)
            tau_cumule_global = tau_cumule_global + tau_sortie_local * (1.0 - tau_cumule_global)

            # Réinitialisation SO3 entre lit 3 et lit 4 (absorption intermédiaire)
            if i == 2:
                y_in['SO3'] = 0.0
                total = sum(y_in.values())
                for k in y_in:
                    y_in[k] /= total

    def _residus(
        self,
        tau: np.ndarray,
        T_K: np.ndarray,
        tau_in: float,
        T_in: float,
        y_in: Dict[str, float],
        lit_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule dτ/dt et dT/dt pour chaque cellule du lit (upwind).

        Schéma upwind (flux positif vers z+) :
            ∂τ/∂t = −u_int·(τ_i − τ_{i−1})/Δz + S_τ
            ∂T/∂t = −u_int·(T_i − T_{i−1})/Δz + S_T
        """
        H_lit = self.H_cat_reel[lit_idx]
        dz    = H_lit / self.N_z
        N     = self.N_z

        # Vitesse interstitielle (m/s) – approximation pression et T moyens
        T_moy_K = float(np.mean(T_K))
        u_int   = self._vitesse_interstitielle(T_moy_K, y_in)

        # Vecteurs des flux amont (upwind)
        tau_up = np.empty(N)
        T_up   = np.empty(N)
        tau_up[0]  = tau_in
        T_up[0]    = T_in
        tau_up[1:] = tau[:N-1]
        T_up[1:]   = T_K[:N-1]

        # Termes advectifs
        adv_tau = u_int * (tau - tau_up) / dz
        adv_T   = u_int * (T_K - T_up)  / dz

        # Termes sources (cinétique)
        S_tau = np.zeros(N)
        S_T   = np.zeros(N)

        C_total_in = self.cond.P_in_Pa / (self.R * T_in)
        F_SO2_in   = (self.Q_in * C_total_in) * y_in['SO2']

        for j in range(N):
            y_loc = self._composition_locale(y_in, float(tau[j]))
            r     = self._vitesse_reaction(y_loc, float(T_K[j]), lit_idx)

            if F_SO2_in > 0:
                S_tau[j] = (
                    (1.0 - self.epsilon) * self.rho_cat * self.A_lit * r / F_SO2_in
                )
            rho_g, cp_g, _ = self._proprietes_gaz(y_loc, float(T_K[j]))
            S_T[j] = (
                (1.0 - self.epsilon) * self.rho_cat * self.A_lit
                * (-self.deltaH_r) * r
                / (self.Q_in * rho_g * cp_g)
            )

        d_tau_dt = -adv_tau + S_tau
        d_T_dt   = -adv_T   + S_T

        return d_tau_dt, d_T_dt

    # ------------------------------------------------------------------
    # Physico-chimie
    # ------------------------------------------------------------------

    def _get_Kp(self, T_K: float) -> float:
        return (10 ** (4958.6 / T_K - 5.133)) / (101325 ** 0.5)

    def _cp_molaire(self, gaz: str, T_K: float) -> float:
        t = T_K / 1000.0
        A, B, C, D, E = self.SHOMATE[gaz]
        return A + B*t + C*t**2 + D*t**3 + E/(t**2)

    def _proprietes_gaz(
        self, y: Dict[str, float], T_K: float
    ) -> Tuple[float, float, float]:
        """Renvoie (ρ [kg/m³], cp [J/(kg·K)], M_moy [kg/mol])."""
        M_moy      = sum(y[g] * self.MOLAR_MASS[g] for g in y)
        rho        = self.cond.P_in_Pa * M_moy / (self.R * T_K)
        Cp_mol_mix = sum(y[g] * self._cp_molaire(g, T_K) for g in y)
        return rho, Cp_mol_mix / M_moy, M_moy

    def _viscosite_gaz(self, T_K: float, y: Dict[str, float]) -> float:
        mu_i = {g: self.mu_ref[g] * (T_K / 273.15)**0.7 for g in self.mu_ref}
        return sum(y[g] * mu_i[g] for g in y if g in mu_i)

    def _vitesse_interstitielle(self, T_K: float, y: Dict[str, float]) -> float:
        """Vitesse interstitielle du gaz [m/s]."""
        return self.Q_in / (self.A_lit * self.epsilon)

    def _perte_charge_ergun(
        self, T_K: float, y: Dict[str, float], H_lit: float
    ) -> Tuple[float, float]:
        """Renvoie (ΔP total [Pa], dP/dz [Pa/m])."""
        rho, _, _ = self._proprietes_gaz(y, T_K)
        mu  = self._viscosite_gaz(T_K, y)
        u   = self.Q_in / self.A_lit
        A   = 150 * mu * (1 - self.epsilon)**2 / (self.D_p**2 * self.epsilon**3)
        B   = 1.75 * rho * (1 - self.epsilon) / (self.D_p * self.epsilon**3)
        dP_dz = A * u + B * u**2
        return dP_dz * H_lit, dP_dz

    def _vitesse_reaction(
        self, y: Dict[str, float], T_K: float, lit_idx: int
    ) -> float:
        P         = self.cond.P_in_Pa
        P_SO2     = y['SO2'] * P
        P_O2      = y['O2']  * P
        P_SO3     = y['SO3'] * P
        C_total   = P / (self.R * T_K)
        C_SO2     = y['SO2'] * C_total
        C_O2      = y['O2']  * C_total

        if P_SO2 <= 0 or P_O2 <= 0:
            return 0.0

        K1 = self.cinetique['K1']['A'] * np.exp(-self.cinetique['K1']['E'] / (self.R * T_K))
        K3 = self.cinetique['K3']['A'] * np.exp(-self.cinetique['K3']['E'] / (self.R * T_K))
        K4 = self.cinetique['K4']['A'] * np.exp(-self.cinetique['K4']['E'] / (self.R * T_K))
        Kp = self._get_Kp(T_K)

        omega = min(P_SO3 / (max(P_SO2, 1e-10) * np.sqrt(max(P_O2, 1e-10)) * Kp), 0.999)
        num   = K1 * C_O2 * C_SO2 * (1.0 - omega**2)
        denom = (1.0 + K3 * P_SO2 + K4 * P_SO3)**2
        return (num / denom) * self.facteur_activite_lits[lit_idx]

    @staticmethod
    def _composition_locale(
        y_in: Dict[str, float], tau: float
    ) -> Dict[str, float]:
        """Composition locale à partir du taux de conversion τ."""
        tau   = max(0.0, min(tau, 0.999))
        denom = 1.0 - 0.5 * tau * y_in['SO2']
        f     = 1.0 / denom if denom > 0 else 1.0
        return {
            'SO2': (1.0 - tau)              * y_in['SO2'] * f,
            'O2':  (y_in['O2'] - 0.5 * tau * y_in['SO2']) * f,
            'SO3': (y_in['SO3'] + tau       * y_in['SO2']) * f,
            'N2':   y_in['N2'] * f,
        }

    # ------------------------------------------------------------------
    # Condition CFL
    # ------------------------------------------------------------------

    def _verifier_cfl(self, dt: float) -> None:
        """Lève une ValueError si la condition CFL est violée."""
        u_int = self.Q_in / (self.A_lit * self.epsilon)
        dz_min = min(H / self.N_z for H in self.H_cat_reel)
        cfl = u_int * dt / dz_min
        if cfl > 1.0:
            raise ValueError(
                f"Condition CFL violée : CFL = {cfl:.3f} > 1. "
                f"Réduire dt ou augmenter N_z. "
                f"dt_max ≈ {dz_min / u_int:.4f} s"
            )

    # ------------------------------------------------------------------
    # Enregistrement de l'historique
    # ------------------------------------------------------------------

    def _enregistrer_historique(self) -> None:
        self.historique_temps.append(self.t_simule)

        # Taux de conversion cumulé global [%]
        tau_cumule = 0.0
        for etat in self.etats:
            tau_local = float(etat.tau[-1])
            tau_cumule = tau_cumule + tau_local * (1.0 - tau_cumule)
        self.historique_tau_sortie.append(tau_cumule * 100.0)

        # Température de sortie de chaque lit [°C]
        T_sorties = [float(etat.T_K[-1]) - 273.15 for etat in self.etats]
        self.historique_T_sortie_lits.append(T_sorties)

    # ------------------------------------------------------------------
    # Construction des résultats finaux
    # ------------------------------------------------------------------

    def _construire_resultats_finaux(self) -> Dict:
        """
        Construit le dictionnaire de résultats comprenant :
          - profils spatiaux (z, τ%, T°C) par lit
          - pertes de charge par lit
          - séries temporelles
          - τ cumulé global final
        """
        profils_z      = []
        profils_tau    = []
        profils_T      = []
        delta_P_liste  = []
        limites_lits   = []
        z_offset       = 0.0

        y_in = self.cond.y0.copy()
        tau_cumule_global = 0.0

        for i in range(self.n_lits):
            H_lit = self.H_cat_reel[i]
            etat  = self.etats[i]
            z     = np.linspace(0, H_lit, self.N_z) + z_offset

            # Taux de conversion cumulé instantané sur le profil
            tau_cumul_profil = [
                (tau_cumule_global + t * (1.0 - tau_cumule_global)) * 100.0
                for t in etat.tau
            ]
            tau_sortie_local = float(etat.tau[-1])
            tau_cumule_global = tau_cumule_global + tau_sortie_local * (1.0 - tau_cumule_global)

            T_C = etat.T_K - 273.15

            # Perte de charge (évaluée à T et composition moyennes)
            T_moy_K = float(np.mean(etat.T_K))
            delta_P, _ = self._perte_charge_ergun(T_moy_K, y_in, H_lit)
            delta_P_liste.append(delta_P)

            profils_z.append(z)
            profils_tau.append(np.array(tau_cumul_profil))
            profils_T.append(T_C)
            limites_lits.append(z_offset + H_lit)
            z_offset += H_lit + 0.3

            y_in = self._composition_locale(y_in, tau_sortie_local)
            if i == 2:
                y_in['SO3'] = 0.0
                total = sum(y_in.values())
                for k in y_in:
                    y_in[k] /= total

        # Concaténation pour compatibilité avec l'ancienne interface
        z_cumul   = np.concatenate(profils_z).tolist()
        tau_cumul = np.concatenate(profils_tau).tolist()
        T_cumul   = np.concatenate([T for T in profils_T]).tolist()

        return {
            # ── Compatibilité ancienne interface ──────────────────
            'delta_P_liste':  delta_P_liste,
            'z_cumul':        z_cumul,
            'tau_cumul':      tau_cumul,
            'T_cumul':        T_cumul,
            'limites_lits':   limites_lits,
            # ── Données temporelles ───────────────────────────────
            'temps':                   list(self.historique_temps),
            'tau_sortie_global_pct':   list(self.historique_tau_sortie),
            'T_sortie_lits_C':         list(self.historique_T_sortie_lits),
            't_simule_s':              self.t_simule,
            # ── Valeurs finales scalaires ─────────────────────────
            'tau_final_pct':           tau_cumule_global * 100.0,
        }

    # ------------------------------------------------------------------
    # Méthode statique : état initial (régime permanent approché)
    # ------------------------------------------------------------------

    @classmethod
    def depuis_regime_permanent(
        cls,
        conditions: Optional[ConditionsEntree] = None,
        N_z: int = 100,
        dt: float = 0.5,
        tol: float = 1e-5,
    ) -> "JumeauNumeriqueConvertisseur":
        """
        Crée un jumeau initialisé au régime permanent.
        Pratique avant d'appliquer une perturbation.
        """
        jumeau = cls(conditions=conditions, N_z=N_z)
        jumeau.etat_stationnaire(tol=tol, dt=dt)
        return jumeau


# ─────────────────────────────────────────────────────────────────────
# Exemple d'utilisation
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Initialisation aux conditions nominales
    cond = ConditionsEntree(Q_in_m3h=624_600.0, P_in_Pa=150_000.0, T_entree_C=420.0)
    jumeau = JumeauNumeriqueConvertisseur(conditions=cond, N_z=100)

    # 2. Simulation dynamique : 600 s avec une perturbation de T à t=300 s
    perturbations = [
        (300.0, {'T_entree_C': 435.0}),   # augmentation de T à t=300 s
        (500.0, {'T_entree_C': 420.0}),   # retour à la normale à t=500 s
    ]

    resultats = jumeau.simuler_transitoire(
        duree_s=600.0,
        dt=0.5,
        perturbations=perturbations,
    )

    print(f"Durée simulée       : {resultats['t_simule_s']:.1f} s")
    print(f"τ final global      : {resultats['tau_final_pct']:.2f} %")
    print(f"ΔP lits (Pa)        : {[f'{dp:.0f}' for dp in resultats['delta_P_liste']]}")
    print(f"T sortie lits (°C)  : {[f'{T:.1f}' for T in resultats['T_sortie_lits_C'][-1]]}")