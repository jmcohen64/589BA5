import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def solve_optimal_free_time(x0, v0, T_guess=100):
    """
    Solves the optimal control problem with free final time for system given by 
    L = 1/2(v(t)^2 + u(t)^2) and u(t) = a(t)^2 + v(t), where x is position, v is velocity, a is acceleration, and u is the control
    Initial state: x0, v0
    Final state: x = 0, v = 0
    """
    # ODE system with time scaling (normalized time τ ∈ [0,1])
    def odefun(tau, y, p):
        T = p[0]
        x1, x2, lam1, lam2 = y

        #optimal control: -1 <= u <= 1
        #smoothing function so that control remains continuous
        #u_opt = -lam2 / (1 + lam2**2)**(1/2)
        #u_opt = -lam2
        u_opt = -lam2 / np.maximum(1.0, (1 + lam2**2)**(1/2))
        #u_opt = -np.tanh(lam2)

        #dynamics
        dx1_dt = x2
        dx2_dt = u_opt - x2
        dlam1_dt = x1
        dlam2_dt = lam1 + u_opt

        print(u_opt, dx1_dt,dx2_dt, dlam1_dt, dlam2_dt)

        # Scale derivatives by T due to d/dτ = T * d/dt
        return T * np.vstack((dx1_dt, dx2_dt, dlam1_dt, dlam2_dt))

    # Boundary conditions with Hamiltonian terminal condition
    def bcfun(ya, yb, p):
        T = p[0]
        x1_T, x2_T, lam1_T, lam2_T = yb
        u_T = -lam2_T

        # Hamiltonian at final time should be 0
        H_T = 1/2 * (x1_T**2 - lam2_T**2) + lam1_T*x2_T - lam2_T*x2_T

        return np.array([
            ya[0] - x0,    # x1(0) = x0
            ya[1] - v0,    # x2(0) = v0
            yb[0],         # x1(T) = 0
            yb[1],         # x2(T) = 0
            H_T            # Hamiltonian condition at T
        ])

    # Time grid (normalized time)
    tau = np.linspace(0, 1,100)

    # Initial guess
    y_guess = np.zeros((4, tau.size))
    """
    y_guess[0] = x0 * (1 - tau)                     # linear guess from x0 to 0
    y_guess[1] = np.linspace(v0, -1, tau.size)       # constant velocity guess
    y_guess[2] = 1 #np.ones_like(tau)                  # λ₁ starting at 1
    y_guess[3] = 1 #np.ones_like(tau)                  # λ₂ starting at 1 (implies u = -1 which gives max negative acceleration toward origin with positive velocity)
    """
    y_guess[0] = x0 * (1 - tau)                     # linear guess from x0 to 0 (position)
    y_guess[1] = np.linspace(-v0, 0, tau.size)      # constant velocity guess (velocity)
    y_guess[2] = 100 * tau                  # λ₁ starting at 1 (position costate)
    y_guess[3] = np.ones_like(tau)                  # λ₂ starting negative, ramping toward 0 (velocity costate)

    #print(type( y_guess[3]))
    #print(len(y_guess[3]))
    # Solve the BVP
    sol = solve_bvp(odefun, bcfun, tau, y_guess, p=[T_guess], max_nodes=100000)

    if sol.status != 0:
        print("Solver did not converge:", sol.message)
        return None

    return sol

# === Run for one initial condition ===
x0 = 100
v0 = 10
sol = solve_optimal_free_time(x0, v0)

# print results
T_opt = sol.p[0]
tau = sol.x
t = tau * T_opt
x1, x2, lam1, lam2 = sol.y
u_opt = -lam2
L_vals = 0.5 * (x1**2 + u_opt**2)
J = np.trapezoid(L_vals, t)

print(f"Optimal time: {T_opt:.4f} seconds")
print(f"Minimum cost: {J:.4f}")

# === Plot results ===
if sol is not None:
    T_opt = sol.p[0]
    t = sol.x * T_opt  # Convert normalized time to real time

    x1, x2, lam1, lam2 = sol.y

    plt.figure(figsize=(10, 6))
    plt.subplot(2,1,1)
    plt.plot(t, x1, label=r"$x(t)$")
    plt.plot(t, x2, label=r"v(t)$")
    plt.ylabel("States")
    plt.legend()

    plt.subplot(2,1,2)
    tau_opt = -lam2
    plt.plot(t, tau_opt, label=r"$u(t)$", color='orange')
    plt.xlabel("Time [s]")
    plt.ylabel("Control input")
    plt.legend()

    plt.suptitle(f"Optimal Trajectory from X₀={x0}, V₀={v0} — T* = {T_opt:.3f} s")
    plt.tight_layout()
    plt.show()
