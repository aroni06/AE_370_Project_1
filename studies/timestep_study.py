"""Time-step study: which h resolves the mass-induced changes?

For each h, the RK4 error in each reported diagnostic is measured against the
same diagnostic computed from IAS15 reference solutions, and compared with the
size of the mass effect itself. The chosen h is the largest step whose error
is at least 100 times smaller than every effect.
"""

import json

import numpy as np

from kepler16 import constants as c
from kepler16.diagnostics import binary_separation, planet_relative
from kepler16.dynamics import unpack
from kepler16.plotting import plt, results_path, save
from kepler16.reference import ias15
from kepler16.simulate import run

H_VALUES = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
DT_OUT = 1.6
SAFETY = 100.0
CASES = {"A": (c.M_P0, "fixed"), "B": (c.M_P0 / 2, "fixed"), "C": (c.M_P0 / 2, "rebuilt")}


def diagnostics(states, masses):
    """Mass-effect diagnostics of cases B and C relative to A."""
    qA, _ = planet_relative(states["A"], masses["A"])
    sA, _ = binary_separation(states["A"])
    d = {}
    for k in ("B", "C"):
        q, _ = planet_relative(states[k], masses[k])
        s, _ = binary_separation(states[k])
        d[f"D_p ({k})"] = q - qA
        d[f"binary change ({k})"] = s - sA
    return d


def main():
    masses = {k: np.array([c.M_A, c.M_B, mp]) for k, (mp, _) in CASES.items()}
    ref_states = {k: ias15(mp, vel, dt_out=DT_OUT)[1] for k, (mp, vel) in CASES.items()}
    ref = diagnostics(ref_states, masses)
    effect = {k: float(np.linalg.norm(v, axis=1).max()) for k, v in ref.items()}

    errors = {k: [] for k in ref}
    errors["position (A)"] = []
    for h in H_VALUES:
        states = {k: run(mp, vel, h=h, save_every=int(round(DT_OUT / h)))[1]
                  for k, (mp, vel) in CASES.items()}
        d = diagnostics(states, masses)
        for k in ref:
            errors[k].append(float(np.linalg.norm(d[k] - ref[k], axis=1).max()))
        r, _ = unpack(states["A"])
        rr, _ = unpack(ref_states["A"])
        errors["position (A)"].append(float(np.linalg.norm(r - rr, axis=2).max()))
        print(f"h={h:<5} " + "  ".join(f"{k}: {v[-1]:.2e}" for k, v in errors.items()))

    ok = [all(errors[k][i] <= effect[k] / SAFETY for k in effect) for i in range(len(H_VALUES))]
    chosen = max(h for h, good in zip(H_VALUES, ok) if good)
    print("effect sizes (AU):", {k: f"{v:.3e}" for k, v in effect.items()})
    print("largest h meeting the 100x margin:", chosen)

    with open(results_path("timestep_study.json"), "w") as f:
        json.dump({"h": H_VALUES, "errors": errors, "effect": effect,
                   "safety_factor": SAFETY, "largest_acceptable_h": chosen}, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    h = np.array(H_VALUES)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "x", "s", "+"]
    for col, mk, k in zip(colors, markers, effect):
        ax.loglog(h, errors[k], mk + "-", color=col, ms=7, label=f"error in {k}")
        ax.axhline(effect[k] / SAFETY, color=col, ls="--", lw=0.8)
    ax.axvline(c.H_CHOSEN, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"time step $h$ (day)")
    ax.set_ylabel("max error over 4576 d (AU)")
    ax.set_title("Diagnostic error vs h (dashed: effect size / 100)")
    ax.text(0.02, 0.97, "binary change: B and C overlap", transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(fontsize=7)
    save(fig, "timestep_study")


if __name__ == "__main__":
    main()
