"""Main experiment: effect of halving (and sweeping) the planet's mass.

Cases, all from the same Table 1 elements and run with h = H_CHOSEN:
  A  m_p = m_p0                    (baseline)
  B  m_p = m_p0 / 2, "fixed"       planet's relative r, v held fixed (draft)
  C  m_p = m_p0 / 2, "rebuilt"     planet's r, v rebuilt for the new mu
Each case is repeated with h / 2 to estimate the numerical error of every
reported number. A sweep over m_p / m_p0 in {0, 0.25, 0.5, 1, 2} follows.
"""

import json

import numpy as np

from kepler16 import constants as c
from kepler16.diagnostics import (binary_separation, conjunction_times, in_plane_phase,
                                  mean_revolution_time, osculating_period, planet_relative)
from kepler16.plotting import plt, results_path, save
from kepler16.simulate import run

H = c.H_CHOSEN
CASES = {"A": (1.0, "fixed"), "B": (0.5, "fixed"), "C": (0.5, "rebuilt")}
SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0]
MIN_PER_DAY = 1440.0


def summarize(t, Y, m):
    q, qd = planet_relative(Y, m)
    s, _ = binary_separation(Y)
    rho = np.linalg.norm(q, axis=1)
    tc = conjunction_times(t, Y)
    return {
        "q": q, "s": s, "conj": tc,
        "rho_min": rho.min(), "rho_max": rho.max(),
        "P_rev": mean_revolution_time(t, in_plane_phase(q, qd)),
        "P_bin": np.polyfit(np.arange(len(tc)), tc, 1)[0],
        "P_osc0": osculating_period(q[0], qd[0], c.G * (c.M_STARS + m[2])),
    }


def compare(base, x):
    """Differences of case x relative to the baseline."""
    n = min(len(base["conj"]), len(x["conj"]))
    dconj = (x["conj"][:n] - base["conj"][:n]) * MIN_PER_DAY
    dP_kep = base["P_rev"] * (x["P_osc0"] / base["P_osc0"] - 1)
    return {
        "Dp_max_AU": np.linalg.norm(x["q"] - base["q"], axis=1).max(),
        "Dp_final_AU": np.linalg.norm(x["q"][-1] - base["q"][-1]),
        "binary_change_max_AU": np.linalg.norm(x["s"] - base["s"], axis=1).max(),
        "rho_min_change_AU": x["rho_min"] - base["rho_min"],
        "rho_max_change_AU": x["rho_max"] - base["rho_max"],
        "P_rev_change_day": x["P_rev"] - base["P_rev"],
        "P_rev_change_kepler_day": dP_kep,
        "P_rev_change_residual_day": x["P_rev"] - base["P_rev"] - dP_kep,
        "P_bin_change_s": (x["P_bin"] - base["P_bin"]) * 86400,
        "conj_shift_final_min": dconj[-1],
        "conj_shift_max_min": dconj[np.argmax(np.abs(dconj))],
        "_dconj": dconj,
    }


def run_cases(h):
    out = {}
    for k, (f, vel) in CASES.items():
        t, Y, m = run(f * c.M_P0, vel, h=h)
        out[k] = summarize(t, Y, m)
    return t, out


def main():
    t, S = run_cases(H)
    t2, S2 = run_cases(H / 2)
    step2 = 2  # h/2 run stores every step, so take every 2nd sample to match t
    for k in S2:
        S2[k]["q"], S2[k]["s"] = S2[k]["q"][::step2], S2[k]["s"][::step2]

    table = {}
    for k in ("B", "C"):
        d, d2 = compare(S["A"], S[k]), compare(S2["A"], S2[k])
        table[k] = {name: {"value": float(v), "num_error": float(abs(v - d2[name]))}
                    for name, v in d.items() if not name.startswith("_")}
        table[k]["_dconj"], table[k]["_Dp"] = d["_dconj"], np.linalg.norm(S[k]["q"] - S["A"]["q"], axis=1)
        table[k]["_Dp_err"] = np.abs(np.linalg.norm(S2[k]["q"] - S2["A"]["q"], axis=1) - table[k]["_Dp"])
    baseline = {"P_rev_day": S["A"]["P_rev"], "P_bin_day": S["A"]["P_bin"],
                "rho_min_AU": S["A"]["rho_min"], "rho_max_AU": S["A"]["rho_max"],
                "P_osc0_day": S["A"]["P_osc0"]}
    print("baseline:", {k: round(v, 6) for k, v in baseline.items()})
    for k in ("B", "C"):
        print(f"case {k}:")
        for name, v in table[k].items():
            if not name.startswith("_"):
                print(f"  {name:28s} {v['value']: .4e}  (+/- {v['num_error']:.1e})")

    # ---- mass sweep -------------------------------------------------------
    sweep = {vel: [] for vel in ("fixed", "rebuilt")}
    base = S["A"]
    for vel in sweep:
        for f in SWEEP:
            x = base if f == 1.0 else summarize(*run(f * c.M_P0, vel, h=H))
            d = compare(base, x)
            sweep[vel].append({"factor": f, **{k: float(v) for k, v in d.items() if not k.startswith("_")}})
            print(f"sweep {vel:7s} m/m0={f:<4}  Dp_max={d['Dp_max_AU']:.3e}  "
                  f"dP={d['P_rev_change_day']: .4f} (kepler {d['P_rev_change_kepler_day']: .4f})  "
                  f"conj shift={d['conj_shift_final_min']: .3f} min")

    with open(results_path("mass_experiment.json"), "w") as fjs:
        json.dump({"h": H, "baseline": baseline,
                   "cases": {k: {n: v for n, v in table[k].items() if not n.startswith("_")}
                             for k in table},
                   "sweep": sweep}, fjs, indent=2)

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots()
    for k, lab in (("B", "B: fixed relative velocity"), ("C", "C: velocity rebuilt")):
        line, = ax.semilogy(t[1:], table[k]["_Dp"][1:], label=lab)
        err = np.maximum.accumulate(table[k]["_Dp_err"])  # running-max envelope
        ax.semilogy(t[1:], np.maximum(err[1:], 1e-16), ":", color=line.get_color(), lw=1.0)
    ax.set_xlabel("time (day)")
    ax.set_ylabel(r"$|D_p(t)|$ (AU)")
    ax.set_title(r"Planet displacement from halving $m_p$ (dotted: numerical error, $h$ vs $h/2$)", fontsize=9)
    ax.legend(fontsize=8)
    save(fig, "results_planet_displacement")

    fig, ax = plt.subplots()
    for k, lab in (("B", "B: fixed relative velocity"), ("C", "C: velocity rebuilt")):
        dc = table[k]["_dconj"]
        ax.plot(S["A"]["conj"][:len(dc)], dc, ".-", ms=3, label=lab)
    ax.set_xlabel("time (day)")
    ax.set_ylabel("conjunction time shift (min)")
    ax.set_title(r"Shift in stellar conjunction (eclipse) times, $m_p \to m_{p,0}/2$")
    ax.legend(fontsize=8)
    save(fig, "results_conjunction_shift")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for vel, mk in (("fixed", "o-"), ("rebuilt", "s-")):
        f = np.array([r["factor"] for r in sweep[vel]])
        a1.plot(f, [r["P_rev_change_day"] for r in sweep[vel]], mk, label=f"{vel}: simulated")
        a1.plot(f, [r["P_rev_change_kepler_day"] for r in sweep[vel]], "k:" if vel == "fixed" else "k--",
                lw=0.8, label=f"{vel}: two-body prediction")
        a2.plot(f, [r["conj_shift_final_min"] for r in sweep[vel]], mk, label=vel)
    a1.set_xlabel(r"$m_p / m_{p,0}$")
    a1.set_ylabel("change in planet revolution time (day)")
    a1.legend(fontsize=7)
    a2.set_xlabel(r"$m_p / m_{p,0}$")
    a2.set_ylabel("conjunction shift at t = 4576 d (min)")
    a2.legend(fontsize=7)
    fig.suptitle("Response to planet mass, relative to the nominal system", fontsize=10)
    save(fig, "results_mass_sweep")


if __name__ == "__main__":
    main()
