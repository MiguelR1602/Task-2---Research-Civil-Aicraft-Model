# =============================================================================
#   RCAM MODEL - TODAS LAS GRÁFICAS VISIBLES
# =============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# === Configuración para Spyder ===
# Esto asegura que las gráficas se abran en ventanas externas sin bloquear
matplotlib.use("Qt5Agg")
plt.ion()   # Modo interactivo ON (permite abrir varias figuras a la vez)


# =============================================================================
# PASO 1 - DEFINICIÓN DEL MODELO RCAM
# =============================================================================
def RCAM_model(X: np.ndarray, U: np.ndarray, rho: float) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 1.1 - CONSTANTES DEL AVIÓN
    # -------------------------------------------------------------------------
    m = 120000.0
    cbar, lt, S, St = 6.6, 24.8, 260.0, 64.0

    Xcg, Ycg, Zcg = 0.23 * cbar, 0.0, 0.10 * cbar
    Xac, Yac, Zac = 0.12 * cbar, 0.0, 0.0
    Xapt1, Yapt1, Zapt1 = 0.0, -7.94, -1.9
    Xapt2, Yapt2, Zapt2 = 0.0, 7.94, -1.9

    g = 9.81
    depsda = 0.25
    deg2rad = np.pi / 180.0
    alpha_L0 = -11.5 * deg2rad
    n = 5.5
    a3, a2, a1, a0 = -768.5, 609.2, -155.2, 15.212
    alpha_switch = 14.5 * deg2rad

    # -------------------------------------------------------------------------
    # PASO 2 - VARIABLES INTERMEDIAS
    # -------------------------------------------------------------------------
    u, v, w = X[0], X[1], X[2]
    Va = np.sqrt(u**2 + v**2 + w**2)
    alpha = np.arctan2(w, u)
    beta = np.arcsin(np.clip(v / Va, -1.0, 1.0))
    Q = 0.5 * rho * Va**2

    wbe_b = np.array([X[3], X[4], X[5]])
    V_b = np.array([X[0], X[1], X[2]])

    # -------------------------------------------------------------------------
    # PASO 3 - COEFICIENTES AERODINÁMICOS
    # -------------------------------------------------------------------------
    if alpha <= alpha_switch:
        CL_wb = n * (alpha - alpha_L0)
    else:
        CL_wb = a3 * alpha**3 + a2 * alpha**2 + a1 * alpha + a0

    epsilon = depsda * (alpha - alpha_L0)
    alpha_t = alpha - epsilon + U[1] + (1.3 * X[4] * lt / Va if Va != 0 else 0)
    CL_t = 3.1 * (St / S) * alpha_t
    CL = CL_wb + CL_t
    CD = 0.13 + 0.07 * (n * alpha + 0.654)**2
    CY = -1.6 * beta + 0.24 * U[2]

    # -------------------------------------------------------------------------
    # PASO 4 - FUERZAS AERODINÁMICAS
    # -------------------------------------------------------------------------
    FA_s = np.array([-CD * Q * S, CY * Q * S, -CL * Q * S])
    C_bs = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [0.0, 1.0, 0.0],
                     [np.sin(alpha), 0.0, np.cos(alpha)]])
    FA_b = np.dot(C_bs, FA_s)

    # -------------------------------------------------------------------------
    # PASO 5 - MOMENTOS AERODINÁMICOS
    # -------------------------------------------------------------------------
    eta11 = -1.4 * beta
    eta21 = -0.59 - (3.1 * (St * lt) / (S * cbar)) * (alpha - epsilon)
    eta31 = (1 - alpha * (180 / (15 * np.pi))) * beta
    eta = np.array([eta11, eta21, eta31])

    dCMdx = (cbar / Va if Va != 0 else 0.0) * np.array([
        [-11.0, 0.0, 5.0],
        [0.0, (-4.03 * (St * lt**2) / (S * cbar**2)), 0.0],
        [1.7, 0.0, -11.5]
    ])

    dCMdu = np.array([
        [-0.6, 0.0, 0.22],
        [0.0, (-3.1 * (St * lt) / (S * cbar)), 0.0],
        [0.0, 0.0, -0.63]
    ])

    CMac_b = eta + np.dot(dCMdx, wbe_b) + np.dot(dCMdu, np.array([U[0], U[1], U[2]]))
    MAac_b = CMac_b * Q * S * cbar

    rcg_b, rac_b = np.array([Xcg, Ycg, Zcg]), np.array([Xac, Yac, Zac])
    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)

    # -------------------------------------------------------------------------
    # PASO 6 - FUERZAS DE MOTOR Y GRAVEDAD
    # -------------------------------------------------------------------------
    F1 = U[3] * m * g
    F2 = U[4] * m * g
    FE_b = np.array([F1 + F2, 0.0, 0.0])

    mew1 = np.array([Xcg - Xapt1, Yapt1 - Ycg, Zcg - Zapt1])
    mew2 = np.array([Xcg - Xapt2, Yapt2 - Ycg, Zcg - Zapt2])
    MEcg_b = np.cross(mew1, np.array([F1, 0, 0])) + np.cross(mew2, np.array([F2, 0, 0]))

    g_b = np.array([
        -g * np.sin(X[7]),
         g * np.cos(X[7]) * np.sin(X[6]),
         g * np.cos(X[7]) * np.cos(X[6])
    ])
    Fg_b = m * g_b

    # -------------------------------------------------------------------------
    # PASO 7 - ECUACIONES DE MOVIMIENTO
    # -------------------------------------------------------------------------
    Ib = m * np.array([[40.07, 0.0, -2.0923],
                       [0.0, 64.0, 0.0],
                       [-2.0923, 0.0, 99.92]])
    invIb = np.linalg.inv(Ib)

    F_b = Fg_b + FE_b + FA_b
    Mcg_b = MAcg_b + MEcg_b

    x0x1x2_dot = (1.0 / m) * F_b - np.cross(wbe_b, V_b)
    x3x4x5_dot = np.dot(invIb, Mcg_b - np.cross(wbe_b, np.dot(Ib, wbe_b)))

    H_phi = np.array([
        [1.0, np.sin(X[6]) * np.tan(X[7]), np.cos(X[6]) * np.tan(X[7])],
        [0.0, np.cos(X[6]), -np.sin(X[6])],
        [0.0, np.sin(X[6]) / np.cos(X[7]), np.cos(X[6]) / np.cos(X[7])]
    ])
    x6x7x8_dot = np.dot(H_phi, wbe_b)

    return np.concatenate((x0x1x2_dot, x3x4x5_dot, x6x7x8_dot))


# =============================================================================
# SIMULADOR E INTEGRADOR
# =============================================================================
def rk4_step(f, t, X, dt, U_sched, rho):
    k1 = f(X, U_sched(t, X), rho)
    k2 = f(X + 0.5 * dt * k1, U_sched(t + 0.5 * dt, X + 0.5 * dt * k1), rho)
    k3 = f(X + 0.5 * dt * k2, U_sched(t + 0.5 * dt, X + 0.5 * dt * k2), rho)
    k4 = f(X + dt * k3, U_sched(t + dt, X + dt * k3), rho)
    return X + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def simulate(f_model, X0, U_sched, rho=1.225, t_final=180, dt_int=0.01, dt_sample=0.5):
    t, X = 0.0, X0.copy().astype(float)
    traj_t, traj_X = [t], [X.copy()]
    next_sample = dt_sample

    while t < t_final:
        X = rk4_step(f_model, t, X, dt_int, U_sched, rho)
        t += dt_int
        if t >= next_sample or t >= t_final:
            traj_t.append(t)
            traj_X.append(X.copy())
            next_sample += dt_sample
    return np.array(traj_t), np.vstack(traj_X)


# =============================================================================
# ESCENARIOS DE CONTROL
# =============================================================================
def base_scheduler(U0): return lambda t, X: U0.copy()

def aileron_pulse_scheduler(U0, t0=10.0, width=2.0, amp_deg=5.0):
    amp = np.deg2rad(amp_deg)
    def sched(t, X):
        U = U0.copy()
        if t0 <= t < t0 + width: U[0] += amp
        return U
    return sched

def engine_shutdown_scheduler(U0, t_fail=10.0, which=1):
    def sched(t, X):
        U = U0.copy()
        if t >= t_fail:
            if which == 1: U[3] = 0.0
            else: U[4] = 0.0
        return U
    return sched


# =============================================================================
# FUNCIÓN DE GRAFICADO
# =============================================================================
def plot_states(t, X, title):
    labels = ["u [m/s]", "v [m/s]", "w [m/s]",
              "p [rad/s]", "q [rad/s]", "r [rad/s]",
              "phi [rad]", "theta [rad]", "psi [rad]"]
    fig, axs = plt.subplots(3, 3, figsize=(12, 9))
    axs = axs.ravel()
    for i in range(9):
        axs[i].plot(t, X[:, i])
        axs[i].set_xlabel("Tiempo [s]")
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()  # <- ahora deja todas las ventanas abiertas


# =============================================================================
# MAIN
# =============================================================================
def main():
    rho = 1.225
    X0 = np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
    U0 = np.array([0.0, -0.1, 0.0, 0.08, 0.08])

    print("\n--- ESCENARIO 1: VUELO BASE ---")
    t1, X1 = simulate(RCAM_model, X0, base_scheduler(U0), rho, 180)
    plot_states(t1, X1, "Escenario 1: Vuelo base")

    print("\n--- ESCENARIO 2: PULSO DE ALERÓN (+5° en t=10s) ---")
    t2, X2 = simulate(RCAM_model, X0, aileron_pulse_scheduler(U0, 10.0, 2.0, 5.0), rho, 180)
    plot_states(t2, X2, "Escenario 2: Pulso de alerón")

    print("\n--- ESCENARIO 3: FALLA MOTOR IZQUIERDO (t=10s) ---")
    t3, X3 = simulate(RCAM_model, X0, engine_shutdown_scheduler(U0, 10.0, which=1), rho, 180)
    plot_states(t3, X3, "Escenario 3: Falla de motor izquierdo")

    print("\n✅ Simulación completada - Todas las gráficas visibles.")


# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    try:
        get_ipython().run_line_magic('matplotlib', 'qt5')
    except Exception:
        pass
    main()


# =============================================================================
# ============================  PUNTO 5: TRIM con PSO  =========================
# =============================================================================
# Este bloque se agrega SIN modificar el código existente. Solo añade
# funciones nuevas y un lanzador automático que NO llama main() otra vez.
# Requiere que en el código original existan: RCAM_model(X, U, rho).
# =============================================================================

import numpy as np

try:
    rng = np.random.default_rng(42)
except Exception:
    rng = np.random.RandomState(42)

def _deg(x):
    return x * np.pi/180.0

# --- Límites de las variables de decisión: [u, v, w, phi, theta, da, de, dr, t]
lb = np.array([ 60.0,  -2.0,  -5.0,  -_deg(5), -_deg(10),  -_deg(15), -_deg(10), -_deg(15), 0.02], dtype=float)
ub = np.array([120.0,   2.0,   5.0,   _deg(5),  _deg(10),   _deg(15),  _deg(10),  _deg(15), 0.20], dtype=float)

def decision_to_state_control(z):
    """Construye X y U para evaluar xdot en la función de costo."""
    z = np.asarray(z, dtype=float)
    u, v, w, phi, theta, da, de, dr, t = z
    # Fijamos rumbo norte psi=0, tasas angulares nulas p=q=r=0
    p = q = r = 0.0
    psi = 0.0
    X = np.array([u, v, w, p, q, r, phi, theta, psi], dtype=float)
    U = np.array([da, de, dr, t, t], dtype=float)  # throttles iguales
    return X, U

def trim_cost(z, rho=1.225):
    # Satura por seguridad
    zc = np.minimum(np.maximum(z, lb), ub)
    X, U = decision_to_state_control(zc)

    # Derivadas de estado con tu modelo RCAM (debe existir en el código original)
    xdot = RCAM_model(X, U, rho)

    # Magnitudes útiles
    u, v, w = X[0], X[1], X[2]
    Va = float(np.sqrt(u*u + v*v + w*w))

    # --- Errores (e_i)
    e_uDot, e_vDot, e_wDot = xdot[0], xdot[1], xdot[2]
    e_pDot, e_qDot, e_rDot = xdot[3], xdot[4], xdot[5]
    e_Va   = Va - 85.0
    e_v    = v              # ~ beta→0
    e_phi  = X[6]           # roll→0
    # Regularización suave de mandos (evita soluciones “raras”)
    da, de, dr, t = U[0], U[1], U[2], U[3]

    # --- Escalas (s_i) para adimensionalizar
    s_uDot = s_vDot = s_wDot = 0.5     # m/s^2
    s_pDot = s_qDot = s_rDot = 0.1     # rad/s^2
    s_Va   = 1.0                        # m/s
    s_v    = 0.2                        # m/s
    s_phi  = _deg(1)                    # rad (~1°)
    s_cmd  = _deg(1)                    # rad

    # --- Pesos (w_i): prioriza equilibrio y Va exacta
    w_acc  = 50.0
    w_ang  = 50.0
    w_Va   = 80.0
    w_lat  = 30.0
    w_roll = 30.0
    w_cmd  =  2.0

    J  = w_acc*((e_uDot/s_uDot)**2 + (e_vDot/s_vDot)**2 + (e_wDot/s_wDot)**2)
    J += w_ang*((e_pDot/s_pDot)**2 + (e_qDot/s_qDot)**2 + (e_rDot/s_rDot)**2)
    J += w_Va*(e_Va/s_Va)**2
    J += w_lat*(e_v/s_v)**2
    J += w_roll*(e_phi/s_phi)**2
    # Regularización de mandos (da, de, dr) ~ suavidad
    J += w_cmd*((da/s_cmd)**2 + (de/s_cmd)**2 + (dr/s_cmd)**2)

    return float(J)

def pso_optimize(cost_fn, n_particles=40, n_iters=120):
    """PSO con parámetros que evolucionan (exploración -> explotación)."""
    dim = lb.size
    Xp = lb + (ub - lb) * rng.random((n_particles, dim))
    Vp = np.zeros_like(Xp)

    # Evaluación inicial
    Jp = np.array([cost_fn(x) for x in Xp])
    Pbest = Xp.copy()
    Jbest = Jp.copy()

    # Global best
    g_idx = int(np.argmin(Jp))
    Gbest = Xp[g_idx].copy()
    Jg = float(Jp[g_idx])

    hist = [Jg]

    for it in range(1, n_iters+1):
        tau = it / n_iters
        w  = 0.9 + (0.4 - 0.9)*tau      # 0.9 -> 0.4
        c1 = 1.8 + (0.5 - 1.8)*tau      # 1.8 -> 0.5
        c2 = 0.5 + (2.0 - 0.5)*tau      # 0.5 -> 2.0

        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))

        Vp = w*Vp + c1*r1*(Pbest - Xp) + c2*r2*(Gbest - Xp)
        Xp = Xp + Vp

        # Restringir a los límites
        Xp = np.minimum(np.maximum(Xp, lb), ub)

        # Evaluar
        Jp = np.array([cost_fn(x) for x in Xp])

        # Actualizar mejores personales
        mask = Jp < Jbest
        if np.any(mask):
            Pbest[mask] = Xp[mask]
            Jbest[mask] = Jp[mask]

        # Actualizar mejor global
        g_idx = int(np.argmin(Jbest))
        if float(Jbest[g_idx]) < Jg:
            Jg = float(Jbest[g_idx])
            Gbest = Pbest[g_idx].copy()

        hist.append(Jg)

    return Gbest, Jg, np.array(hist, dtype=float)

def run_trim_search(rho=1.225, t_check=60.0):
    import matplotlib.pyplot as plt
    print("\\n--- PUNTO 5: Búsqueda de trim con PSO ---")
    z_opt, J_opt, hist = pso_optimize(lambda z: trim_cost(z, rho=rho),
                                      n_particles=40, n_iters=120)
    Xtrim, Utrim = decision_to_state_control(z_opt)
    print(f"J* = {J_opt:.6f}")
    print("Xtrim =", Xtrim)
    print("Utrim =", Utrim)

    # ÚNICA GRÁFICA: Convergencia del PSO (J vs iteración)
    fig = plt.figure()
    plt.plot(hist, linewidth=2.0)
    plt.xlabel("Iteración")
    plt.ylabel("Costo J")
    plt.title("Convergencia PSO")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return z_opt, J_opt, hist

# ========================= EJECUCIÓN AUTOMÁTICA ===============================
# Este bloque corre automáticamente SOLO el Punto 5 (PSO) para evitar duplicar
# las gráficas de los Puntos 1–4, que ya son ejecutadas por el main() original.
# =============================================================================
if __name__ == "__main__":
    try:
        print("\\n--- PUNTO 5: TRIM CON PSO ---")
        z_opt, J_opt, hist = run_trim_search(rho=1.225, t_check=60.0)
    except Exception as e:
        print("Error en Punto 5:", e)



