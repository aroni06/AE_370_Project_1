"""Verification 1: RK4 against the exact two-body (Kepler) solution.

With m_p = 0 the planet exerts no force, so the two stars follow an exact
Keplerian orbit about their barycenter. The RK4 error in the stellar
positions is measured against that analytic solution for a range of h, on a
short intervals (1 and 5 binary orbits) and on the full study interval.
"""

import json

import numpy as np

from kepler16 import constants as c
from kepler16.dynamics import unpack
from kepler16.orbits import relative_state
from kepler16.plotting import plt, results_path, save
from kepler16.simulate import run

H_VALUES = [1.6, 0.8, 0.4, 0.2, 0.1, 0.05]
DT_OUT = 1.6
INTERVALS = {"one_orbit": 41.6, "five_orbits": 204.8, "long": c.T_FINAL}   # ~1 and ~5 binary orbits


def exact_star_positions(t):
    r_ba, _ = relative_state(c.BINARY, c.G * c.M_STARS, t)
    return -c.M_B / c.M_STARS * r_ba, c.M_A / c.M_STARS * r_ba


def max_error(h, t_final):
    t, Y, _ = run(0.0, "rebuilt", h=h, t_final=t_final, save_every=int(round(DT_OUT / h)))
    r, _ = unpack(Y)
    rA, rB = exact_star_positions(t)
    err = np.maximum(np.linalg.norm(r[:, 0] - rA, axis=1), np.linalg.norm(r[:, 1] - rB, axis=1))
    return err.max()


def main():
    out = {}
    for name, T in INTERVALS.items():
        E = np.array([max_error(h, T) for h in H_VALUES])
        p = np.log2(E[:-1] / E[1:])
        out[name] = {"T_days": T, "h": H_VALUES, "E": E.tolist(), "p_obs": p.tolist()}
        print(f"{name} interval ({T} d)")
        for h, e, pp in zip(H_VALUES, E, [np.nan, *p]):
            print(f"  h={h:<5} E={e:.3e} AU  p_obs={pp:.2f}")

    with open(results_path("verify_two_body.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots()
    h = np.array(H_VALUES)
    for name, mk, lab in (("one_orbit", "o", "1 binary orbit (41.6 d)"),
                          ("five_orbits", "^", "5 binary orbits (204.8 d)"),
                          ("long", "s", "full interval (4576 d)")):
        ax.loglog(h, out[name]["E"], mk + "-", label=lab)
    ref = out["one_orbit"]["E"][2]
    ax.loglog(h, ref * (h / h[2]) ** 4, "k--", lw=0.8, label=r"$\propto h^4$")
    ref = out["long"]["E"][2]
    ax.loglog(h, ref * (h / h[2]) ** 5, "k:", lw=0.8, label=r"$\propto h^5$")
    ax.set_xlabel(r"time step $h$ (day)")
    ax.set_ylabel(r"max stellar position error (AU)")
    ax.set_title("RK4 vs exact two-body solution ($m_p = 0$)")
    ax.legend(fontsize=8)
    save(fig, "verify_two_body")


if __name__ == "__main__":
    main()
