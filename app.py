import numpy as np
from flask import Flask, jsonify, render_template, request
from scipy.integrate import solve_ivp

app = Flask(__name__)

# L = 2000  # Egg laying rate (eggs/day)
# W = 5000  # Number of hive bees for 50% egg survival
# R_b = 0.25  # Baseline recruitment rate (per day)
# alpha_f = 0.25  # Additional recruitment in absence of food (per day)
# alpha_F = 0.75  # Effect of excess foragers on recruitment (per day)
# m = 0.14  # Natural death rate of foragers (per day)
# m_w = 0.0056  # Natural death rate of foragers and hive bees in winter (per day)
# b = 500  # Mass of food stored for 50% egg survival (g)
# c = 0.1  # Food gathered per day per forager (g/day)

# begin outline for write up, dw about order or anything specific
# gh repo for code


def beta_function(t, kappa, beta_value):
    if t < kappa:
        return 0
    return beta_value


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/simulate", methods=["POST"])
def simulate():
    # Parameters
    L = float(request.form.get("L"))
    W = float(request.form.get("W"))
    R_b = float(request.form.get("R_b"))
    alpha_f = float(request.form.get("alpha_f"))
    alpha_F = float(request.form.get("alpha_F"))
    m = float(request.form.get("m"))
    m_w = float(request.form.get("m_w"))
    b = float(request.form.get("b"))
    c = float(request.form.get("c"))
    c_I = float(request.form.get("c_I"))

    gamma = float(request.form.get("gamma"))
    gamma_I = float(request.form.get("gamma_I"))
    dH = float(request.form.get("dH"))
    dF = float(request.form.get("dF"))
    beta_value = float(request.form.get("beta"))
    kappa = float(request.form.get("kappa"))

    # Initial conditions
    # H_S0 = 3.4e4  # Initial susceptible hive bee population
    # H_I0 = 100  # Initial infected hive bee population
    # F_S0 = 1.2e4  # Initial susceptible forager bee population
    # F_I0 = 100  # Initial infected forager bee population
    # f0 = 1000  # Initial amount of food

    H_S0 = float(request.form.get("H_S0"))
    H_I0 = float(request.form.get("H_I0"))
    F_S0 = float(request.form.get("F_S0"))
    F_I0 = float(request.form.get("F_I0"))
    f0 = float(request.form.get("f0"))

    # Time points (in days)
    days = 450
    # days = 350
    t = np.linspace(0, days, days * 4)

    # Differential equations
    def deriv(t, y, L, W, Rb, af, aF, m, b, c, gamma, dH, dF, m_w, beta_value):
        H_S, H_I, F_S, F_I, f = y

        is_active_season = t < 150 or t > 250
        beta = beta_function(t, kappa, beta_value)

        if t > 130:
            beta = 0.00001
        #     beta = 0.000005

        N = H_S + H_I + F_S + F_I

        S = (H_S + H_I) / (W + H_S + H_I) * (f / (b + f))
        R = Rb + af * (b / (b + f)) - aF * (F_I + F_S) / N

        if is_active_season:
            dH_S_dt = L * S - H_S * R - (beta * H_I + beta * F_I) * H_S
            dH_I_dt = (beta * H_I + beta * F_I) * H_S - H_I * R - dH * H_I
            dF_S_dt = H_S * R - m * F_S - (beta * H_I + beta * F_I) * F_S
            dF_I_dt = (beta * H_I + beta * F_I) * F_S - m * F_I - dF * F_I
            df_dt = c * (F_S + F_I) - gamma * N
        else:  # is winter
            dH_S_dt = -m_w * H_S - (beta * H_I + beta * F_I) * H_S
            dF_S_dt = -m_w * F_S - (beta * H_I + beta * F_I) * F_S
            dH_I_dt = (beta * H_I + beta * F_I) * H_S - (m_w + dH) * H_I
            dF_I_dt = (beta * H_I + beta * F_I) * F_S - (m_w + dF) * F_I
            df_dt = -gamma * N - gamma * L * S

        # make sure no negative populations
        dH_S_dt = max(dH_S_dt, -H_S)
        dH_I_dt = max(dH_I_dt, -H_I)
        dF_S_dt = max(dF_S_dt, -F_S)
        dF_I_dt = max(dF_I_dt, -F_I)
        df_dt = max(df_dt, -f)

        if t > 80:
            if H_S <= 1:
                dH_S_dt = 0
                H_S = 0

            if H_I <= 1:
                dH_I_dt = 0
                H_I = 0

            if F_S <= 1:
                dF_S_dt = 0
                F_S = 0

            if F_I <= 1:
                dF_I_dt = 0
                F_I = 0

            if f <= 1:
                df_dt = 0
                f = 0

        # if t >= 130 and t <= 132:
        #     df_dt = -f + f0

        return [dH_S_dt, dH_I_dt, dF_S_dt, dF_I_dt, df_dt]

    # Initial condition vector
    y0 = [H_S0, H_I0, F_S0, F_I0, f0]

    # Integrate the differential equations over the time grid, t.
    solution = solve_ivp(
        deriv,
        [t[0], t[-1]],
        y0,
        t_eval=t,
        method="RK45",
        args=(L, W, R_b, alpha_f, alpha_F, m, b, c, gamma, dH, dF, m_w, beta_value),
    )

    if solution.success:
        H_S, H_I, F_S, F_I, f = solution.y

        return jsonify(
            {
                "data": {
                    "time": t.tolist(),
                    "H_S": H_S.tolist(),
                    "H_I": H_I.tolist(),
                    "F_S": F_S.tolist(),
                    "F_I": F_I.tolist(),
                    "f": f.tolist(),
                }
            }
        )

    return jsonify({"error": "Simulation failed."})


if __name__ == "__main__":
    app.run(debug=True, port=1827)
