"""Verification 3: RK4 against REBOUND's IAS15 for the full three-body problem.

IAS15 is an adaptive 15th-order integrator with its own force calculation.
Both codes start from the identical state, masses, units and G, and IAS15 is
stopped exactly at each RK4 output time (no interpolation).
"""

import json

import numpy as np

from kepler16 import constants as c
from kepler16.dynamics import unpack
from kepler16.plotting import plt, results_path, save
from kepler16.reference import ias15
from kepler16.simulate import run

H_VALUES = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
DT_OUT = 1.6
INTERVALS = {"one_orbit": 41.6, "long": c.T_FINAL}


def position_error(Y, Yref):
    """Eq. (22): max over output times of the max over bodies."""
    r, _ = unpack(Y)
    rr, _ = unpack(Yref)
    return np.linalg.norm(r - rr, axis=2).max()


def main():
    out = {}
    for name, T in INTERVALS.items():
        _, Yref = ias15(c.M_P0, t_final=T, dt_out=DT_OUT)
        E = np.array([position_error(run(c.M_P0, h=h, t_final=T,
                                         save_every=int(round(DT_OUT / h)))[1], Yref)
                      for h in H_VALUES])
        p = np.log2(E[:-1] / E[1:])
        out[name] = {"T_days": T, "h": H_VALUES, "E": E.tolist(), "p_obs": p.tolist()}
        print(f"{name} ({T} d)")
        for h, e, pp in zip(H_VALUES, E, [np.nan, *p]):
            print(f"  h={h:<5} E={e:.3e} AU  p_obs={pp:.2f}")

    with open(results_path("verify_rebound.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots()
    h = np.array(H_VALUES)
    ax.loglog(h, out["one_orbit"]["E"], "o-", label="1 binary orbit (41.6 d)")
    ax.loglog(h, out["long"]["E"], "s-", label="full interval (4576 d)")
    e = out["one_orbit"]["E"][2]
    ax.loglog(h, e * (h / h[2]) ** 4, "k--", lw=0.8, label=r"$\propto h^4$")
    e = out["long"]["E"][2]
    ax.loglog(h, e * (h / h[2]) ** 5, "k:", lw=0.8, label=r"$\propto h^5$")
    ax.set_xlabel(r"time step $h$ (day)")
    ax.set_ylabel("max position error vs IAS15 (AU)")
    ax.set_title("RK4 vs REBOUND IAS15, three bodies, nominal mass")
    ax.legend(fontsize=8)
    save(fig, "verify_rebound")


if __name__ == "__main__":
    main()
