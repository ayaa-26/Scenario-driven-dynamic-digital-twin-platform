# =====================================================================
# model/chaudiere.py  —  Modèle DYNAMIQUE de la chaudière (WHB)
# Coefficients d'échange h_gaz et h_eau VARIABLES 
#
# h_gaz : corrélation de Zukauskas (faisceau tubulaire, écoulement externe)
# h_eau : corrélation de Cooper    (pool boiling nucléé, fonction de P_r et M)
# Propriétés thermophysiques du gaz SO2/SO3/N2 estimées par polynômes en T.
# =====================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# ── Géométrie ─────────────────────────────────────────────────────────
N_TUBES      = 1122
D_TUBE_EXT   = 0.033
D_TUBE_INT   = 0.027
L_CHAUDIERE  = 13.6
N_CELLS      = 20
DZ           = L_CHAUDIERE / N_CELLS

A_EXT_CELL   = N_TUBES * np.pi * D_TUBE_EXT * DZ
A_INT_CELL   = N_TUBES * np.pi * D_TUBE_INT * DZ

EP_TUBE      = (D_TUBE_EXT - D_TUBE_INT) / 2
VOL_METAL    = N_TUBES * np.pi * ((D_TUBE_EXT/2)**2 - (D_TUBE_INT/2)**2) * DZ
M_METAL_CELL = 7800.0 * VOL_METAL   # masse métal par cellule
CP_METAL     = 500.0                # J/(kg·K)
LAMBDA_ACIER = 50.0                 # W/(m·K)

# ── Conditions vapeur saturée ─────────────────────────────────────────
P_VAPEUR     = 64.76e5      # Pa
T_SAT        = 280.6 + 273.15   # K
T_TARGET_CONV= 420.0            # °C (consigne sortie mélange)

_PRESSURES   = np.array([1,10,20,30,40,50,60,70,80,90,100])
_DH_TABLE    = np.array([2257,2015,1890,1804,1713,1640,1578,1523,1473,1428,1387]) * 1e3

def get_deltaH_vap(P_bar):
    return float(np.interp(P_bar, _PRESSURES, _DH_TABLE))

P_VAPEUR_BAR = P_VAPEUR / 1e5
DELTA_H_VAP  = get_deltaH_vap(P_VAPEUR_BAR)

CP_EAU_LIQ   = 4800.0        # J/(kg·K)
T_EAU_ALIM   = 230.0         # °C
Q_SENS_UNIT  = CP_EAU_LIQ * (T_SAT - 273.15 - T_EAU_ALIM)   # J/kg

# ── Géométrie calandre (pour Zukauskas) ───────────────────────────────
# Diamètre de calandre estimé 
D_CALANDRE   = 2.0          # m
PITCH_T      = D_CALANDRE / np.sqrt(N_TUBES)   # pas transversal estimé (m)
PITCH_L      = PITCH_T                          # disposition carrée
RATIO_SECTION = (PITCH_T - D_TUBE_EXT) / PITCH_T   # fraction libre
N_ROWS       = max(1, int(np.sqrt(N_TUBES)))   # nombre de rangées estimé

# =====================================================================
# PROPRIÉTÉS THERMOPHYSIQUES DU GAZ
# =====================================================================

def gas_properties(T_K: float) -> dict:
    """
    Propriétés thermophysiques d'un mélange SO2/SO3/N2 typique.
    Plage 400–1200 K.
    """
    T = np.clip(T_K, 400.0, 1200.0)

    # Viscosité dynamique [Pa·s]  
    mu = 1.18e-5 + 9.5e-9 * (T - 300.0)

    # Conductivité thermique [W/(m·K)]
    lambda_g = 0.017 + 5.5e-5 * (T - 300.0)

    
    cp_g = 1190.0 + 0.15 * (T - 600.0)

    # Nombre de Prandtl
    Pr = mu * cp_g / lambda_g

    # Densité [kg/m³] (P ≈ 1.1 bar, M_mix ≈ 40 g/mol)
    rho = (40e-3 * 1.1e5) / (8.314 * T)

    return {"mu": mu, "lambda_g": lambda_g, "Pr": Pr, "rho": rho, "cp_g": cp_g}

# =====================================================================
# CORRÉLATION h_gaz — Zukauskas (faisceau tubulaire, écoulement externe)
# =====================================================================

def h_gaz_zukauskas(T_gaz_K: float, mdot_gaz: float) -> float:
    """
    Coefficient convectif côté gaz (calandre) par Zukauskas.
    """
    props = gas_properties(T_gaz_K)
    mu    = props["mu"]
    lam   = props["lambda_g"]
    Pr    = props["Pr"]
    rho   = props["rho"]

    # Section de passage minimale
    A_calandre = np.pi * (D_CALANDRE / 2)**2
    A_min      = A_calandre * RATIO_SECTION
    v_max      = mdot_gaz / (rho * max(A_min, 0.01))

    Re = rho * v_max * D_TUBE_EXT / mu
    Re = max(Re, 1.0)

    # Coefficients C et m selon Re
    if Re < 100:
        C, m = 0.80, 0.40
    elif Re < 1000:
        C, m = 0.27, 0.63
    elif Re < 2e5:
        C, m = 0.021, 0.84
    else:
        C, m = 0.016, 0.84

    # Correction pour nombre de rangées
    F_N = 1.0 if N_ROWS >= 20 else (0.64 + 0.36 * (1 - np.exp(-N_ROWS / 5)))

    # Prandtl à la paroi 
    Pr_w = gas_properties(T_SAT + 50.0)["Pr"]

    Nu = C * F_N * (Re**m) * (Pr**0.36) * ((Pr / max(Pr_w, 0.01))**0.25)
    h = Nu * lam / D_TUBE_EXT
    return float(np.clip(h, 50.0, 300.0))

# =====================================================================
# CORRÉLATION h_eau — Cooper (pool boiling nucléé)
# =====================================================================

M_EAU        = 18.015        # kg/kmol
P_CRIT_EAU   = 220.64e5      # Pa

def h_eau_cooper(q_flux: float) -> float:
    """
    Coefficient d'ébullition nucléée côté eau par Cooper (1984).
    q_flux : flux de chaleur local [W/m²].
    """
    Rp   = 1.0                # μm (rugosité standard)
    P_r  = P_VAPEUR / P_CRIT_EAU
    q    = max(abs(q_flux), 1e4)   # W/m²

    exponent_Pr = 0.12 - 0.2 * np.log10(max(Rp, 1e-6))
    h = (55.0
         * (P_r ** exponent_Pr)
         * ((-np.log10(P_r)) ** (-0.55))
         * (M_EAU ** (-0.5))
         * (q ** 0.67))
    return float(np.clip(h, 3000.0, 15000.0))

# =====================================================================
# PERTE DE CHARGE (Churchill) – pour les extras
# =====================================================================

def pressure_drop_churchill(mdot_gaz: float, T_moy_K: float, M_mix: float) -> tuple:
    """
    Perte de charge côté gaz (écoulement interne dans les tubes).
    Retourne (dP_Pa, v_ms, Re).
    """
    props = gas_properties(T_moy_K)
    mu = props["mu"]
    rho = props["rho"]
    cp = props["cp_g"]   

    A_tube = np.pi/4 * D_TUBE_INT**2
    mdot_tube = mdot_gaz / N_TUBES
    v = mdot_tube / (rho * A_tube)
    Re = rho * v * D_TUBE_INT / mu
    Re = max(Re, 1.0)

    eps_D = 4.6e-5 / D_TUBE_INT   # rugosité relative
    A_c = (2.457 * np.log(1.0 / ((7.0/Re)**0.9 + 0.27*eps_D)))**16
    B_c = (37530.0 / Re)**16
    f = 8.0 * ((8.0/Re)**12 + (A_c + B_c)**(-1.5))**(1.0/12.0)
    dP = f * (L_CHAUDIERE / D_TUBE_INT) * 0.5 * rho * v**2
    return dP, v, Re

# =====================================================================
# PROFIL GAZ (quasi-stationnaire, par cellules)
# =====================================================================

def gaz_profile(Tw_arr, T_gaz_in_K, mdot_gaz, cp_kg):
    """
    Calcule la température de sortie du gaz pour chaque cellule,
    étant donnée la température de paroi Tw_arr.
    Utilise l'efficacité d'une cellule: Tg_out = Tw + (Tg_in - Tw)*exp(-NTU)
    """
    Tg = np.zeros(N_CELLS)
    Tg_prev = T_gaz_in_K
    for i in range(N_CELLS):
        T_loc = 0.5 * (Tg_prev + Tw_arr[i])
        h_g = h_gaz_zukauskas(T_loc, mdot_gaz)
        UA_cell = h_g * A_EXT_CELL
        NTU_cell = UA_cell / (max(mdot_gaz * cp_kg, 1e-6))
        epsilon = 0.85 * (1.0 - np.exp(-NTU_cell))
        Tg_out = Tw_arr[i] + (Tg_prev - Tw_arr[i]) * (1 - epsilon)
        Tg[i] = max(Tg_out, T_SAT)
        Tg_prev = Tg[i]
    return Tg

# =====================================================================
# MODÈLE DYNAMIQUE (EDO sur la paroi)
# =====================================================================

def solve_boiler_dynamic(T_in_gaz, flux_gaz, M_mix, cp_kg, t_sim=3600.0):
    """
    Version dynamique avec intégration temporelle jusqu'à l'équilibre.
    Retourne (T_cold_exit_C, bypass_pct, mdot_vapeur_th, power_mw)
    """
    n_tot = sum(flux_gaz.values())
    mdot_gaz = n_tot * M_mix
    T_gaz_in_K = T_in_gaz + 273.15

    # Estimation initiale de la paroi
    T_wall_init = np.full(N_CELLS, (T_gaz_in_K + T_SAT) / 2.0)

    # Fonction dérivée de la paroi
    def dwall_dt(t, Tw):
        Tg_arr = gaz_profile(Tw, T_gaz_in_K, mdot_gaz, cp_kg)
        dTw = np.zeros(N_CELLS)
        for i in range(N_CELLS):
            T_film = 0.5 * (Tg_arr[i] + Tw[i])
            h_g = h_gaz_zukauskas(T_film, mdot_gaz)
            Qgw = h_g * A_EXT_CELL * (Tg_arr[i] - Tw[i])

            q_flux = max(Qgw / A_INT_CELL, 0.0)
            h_e = h_eau_cooper(q_flux)
            Qwe = h_e * A_INT_CELL * max(Tw[i] - T_SAT, 0.0)

            dTw[i] = (Qgw - Qwe) / (M_METAL_CELL * CP_METAL)
        return dTw

    # Intégration sur t_sim secondes (on prend le dernier point)
    sol = solve_ivp(dwall_dt, (0.0, t_sim), T_wall_init,
                    method='RK45', rtol=1e-4, atol=1e-3,
                    max_step=30.0, dense_output=False)
    Tw_final = sol.y[:, -1]
    Tg_final = gaz_profile(Tw_final, T_gaz_in_K, mdot_gaz, cp_kg)

    T_cold_exit_K = Tg_final[-1]
    T_cold_exit = T_cold_exit_K - 273.15

    # Calcul du bypass pour atteindre T_TARGET_CONV = 420°C
    if T_in_gaz > T_TARGET_CONV > T_cold_exit:
        ratio_bypass = (T_TARGET_CONV - T_cold_exit) / (T_in_gaz - T_cold_exit)
    else:
        ratio_bypass = 0.0
    ratio_bypass = np.clip(ratio_bypass, 0.0, 1.0)

    mdot_boiler = mdot_gaz * (1.0 - ratio_bypass)
    Q_recup = max(mdot_boiler * cp_kg * (T_gaz_in_K - T_cold_exit_K), 0.0)
    mdot_vapeur_kg_s = Q_recup / (Q_SENS_UNIT + DELTA_H_VAP)
    mdot_vapeur_th = mdot_vapeur_kg_s * 3.6   # t/h (1 kg/s = 3.6 t/h)
    power_mw = Q_recup / 1e6

    # ========== AJOUT DES AFFICHAGES ==========
    print("\n=== RÉSULTATS CHAUDIÈRE ===")
    print(f"Température de sortie des gaz : {T_cold_exit:.2f} °C")
    print(f"Taux de bypass                 : {ratio_bypass*100:.2f} %")
    print(f"Production de vapeur           : {mdot_vapeur_th:.2f} t/h")
    print(f"Puissance thermique récupérée  : {power_mw:.2f} MW")
    print("==============================\n")
    # ========================================

    return T_cold_exit, ratio_bypass * 100.0, mdot_vapeur_th, power_mw

# =====================================================================
# FONCTION PRINCIPALE (interface avec le reste du projet)
# =====================================================================

def solve_boiler_and_bypass(T_in_gaz, flux_gaz, M_mix, cp_kg):
    """
    Interface standadrd.
    Retourne (T_cold_exit, bypass_pct, mdot_vapeur_th, power_mw)
    """
    return solve_boiler_dynamic(T_in_gaz, flux_gaz, M_mix, cp_kg)


T_TARGET_CONV = 420.0
def simulate_step_response(T_in_gaz_initial, T_in_gaz_final, t_step, flux_gaz, M_mix, cp_kg, t_total=3600):
    """
    Simule la réponse de la chaudière à un échelon de température d'entrée des gaz.
    
    Paramètres:
        T_in_gaz_initial : float (°C) avant l'échelon
        T_in_gaz_final    : float (°C) après l'échelon
        t_step            : float (s) instant de l'échelon
        flux_gaz, M_mix, cp_kg : paramètres habituels
        t_total           : durée totale de simulation (s)
    """
    from scipy.integrate import solve_ivp
    
    n_tot = sum(flux_gaz.values())
    mdot_gaz = n_tot * M_mix
    T_sat_K = 280.6 + 273.15
    
    # État initial (régime permanent à T_in_gaz_initial)
    # Ici, on fait une pré-simulation sur 3600s à T_in_gaz_initial pour obtenir l'état initial.
    T_gaz_in_initial_K = T_in_gaz_initial + 273.15
    
    def dwall_dt(t, Tw, T_gaz_in_K):
        Tg_arr = gaz_profile(Tw, T_gaz_in_K, mdot_gaz, cp_kg)
        dTw = np.zeros(N_CELLS)
        for i in range(N_CELLS):
            T_film = 0.5 * (Tg_arr[i] + Tw[i])
            h_g = h_gaz_zukauskas(T_film, mdot_gaz)
            Qgw = h_g * A_EXT_CELL * (Tg_arr[i] - Tw[i])
            q_flux = max(Qgw / A_INT_CELL, 0.0)
            h_e = h_eau_cooper(q_flux)
            Qwe = h_e * A_INT_CELL * max(Tw[i] - T_sat_K, 0.0)
            dTw[i] = (Qgw - Qwe) / (M_METAL_CELL * CP_METAL)
        return dTw
    
    # Pré-simulation pour atteindre l'état permanent initial
    Tw_initial_guess = np.full(N_CELLS, (T_gaz_in_initial_K + T_sat_K)/2.0)
    sol_init = solve_ivp(lambda t, Tw: dwall_dt(t, Tw, T_gaz_in_initial_K),
                         (0, 3600), Tw_initial_guess, method='RK45', max_step=30)
    Tw_initial = sol_init.y[:, -1]

if __name__ == "__main__":
    # Données issues de la sortie du four
    T_entree_chaudiere = 1094.11   # °C
    n_total = 4510.25   # mol/s
    y_SO2 = 0.10991
    y_O2  = 0.09978
    y_SO3 = 0.00150
    y_N2  = 0.78882
    flux_gaz = {
        'SO2': y_SO2 * n_total,
        'O2':  y_O2  * n_total,
        'SO3': y_SO3 * n_total,
        'N2':  y_N2  * n_total,
    }
    M_mix = 0.03245   # kg/mol (32.45 g/mol)
    cp_kg = 1190

    # Appel du modèle chaudière
    solve_boiler_and_bypass(T_entree_chaudiere, flux_gaz, M_mix, cp_kg)