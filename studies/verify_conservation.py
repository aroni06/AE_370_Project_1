"""Verification 2: conservation of energy, angular momentum and momentum.

The isolated point-mass system conserves all three exactly. RK4 conserves
linear momentum to roundoff (it is a linear invariant) but not energy or
angular momentum, whose errors must shrink as h decreases.
"""

import json

import numpy as np

from kepler16 import constants as c
from kepler16.dynamics import angular_momentum, energy, unpack
from kepler16.plotting import plt, results_path, save
from kepler16.simulate import run

H_VALUES = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
DT_OUT = 1.6


def main():
    rows, histories = [], {}
    for h in H_VALUES:
        t, Y, m = run(c.M_P0, h=h, save_every=int(round(DT_OUT / h)))
        E, L = energy(Y, m), angular_momentum(Y, m)
        dE = np.abs(E / E[0] - 1)
        dL = np.linalg.norm(L - L[0], axis=1) / np.linalg.norm(L[0])
        _, v = unpack(Y)
        P = np.linalg.norm(np.einsum("i,tij->tj", m, v), axis=1)
        rows.append({"h": h, "max_rel_energy_err": dE.max(), "max_rel_L_err": dL.max(),
                     "max_momentum": P.max()})
        histories[h] = (t, dE)
        print(f"h={h:<5} |dE/E|={dE.max():.3e}  |dL|/|L|={dL.max():.3e}  |P|max={P.max():.1e}")

    for a, b in zip(rows[:-1], rows[1:]):
        b["p_energy"] = float(np.log2(a["max_rel_energy_err"] / b["max_rel_energy_err"]))
    with open(results_path("verify_conservation.json"), "w") as f:
        json.dump(rows, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    h = np.array(H_VALUES)
    ax1.loglog(h, [r["max_rel_energy_err"] for r in rows], "o-", label="energy")
    ax1.loglog(h, [r["max_rel_L_err"] for r in rows], "s-", label="angular momentum")
    e2 = rows[3]["max_rel_energy_err"]
    ax1.loglog(h, e2 * (h / h[3]) ** 5, "k:", lw=0.8, label=r"$\propto h^5$")
    ax1.set_xlabel(r"time step $h$ (day)")
    ax1.set_ylabel("max relative error over 4576 d")
    ax1.legend(fontsize=8)
    for hh in (0.4, 0.2, 0.1):
        t, dE = histories[hh]
        ax2.semilogy(t[1:], dE[1:], label=f"$h$ = {hh} d")
    ax2.set_xlabel("time (day)")
    ax2.set_ylabel(r"$|E(t)/E(0) - 1|$")
    ax2.legend(fontsize=8)
    fig.suptitle("Conservation checks, nominal planet mass", fontsize=10)
    save(fig, "verify_conservation")


if __name__ == "__main__":
    main()
