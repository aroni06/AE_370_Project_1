"""Independent reference solutions from REBOUND's IAS15 integrator.

REBOUND is used ONLY to produce benchmark solutions for verification. It
never advances the RK4 solution used in the results.
"""

import os

import numpy as np

from . import constants as c
from .orbits import initial_state
from .plotting import results_path


def ias15(m_p, velocity="fixed", t_final=c.T_FINAL, dt_out=1.6, cache=True):
    """IAS15 solution sampled at t = 0, dt_out, 2 dt_out, ..., t_final.

    Results are cached in results/ so later studies reuse them.
    """
    fname = results_path(f"ias15_{velocity}_{m_p / c.M_P0:.3f}mp0_{t_final:g}d_{dt_out:g}.npz")
    if cache and os.path.exists(fname):
        d = np.load(fname)
        return d["t"], d["Y"]

    import rebound

    m, y0 = initial_state(m_p, velocity)
    sim = rebound.Simulation()
    sim.G = c.G
    sim.integrator = "ias15"
    for i in range(3):
        x, y, z, vx, vy, vz = y0[6 * i:6 * i + 6]
        sim.add(m=m[i], x=x, y=y, z=z, vx=vx, vy=vy, vz=vz)

    t = np.arange(int(round(t_final / dt_out)) + 1) * dt_out
    Y = np.empty((len(t), 18))
    for k, tk in enumerate(t):
        sim.integrate(tk, exact_finish_time=1)
        for i, p in enumerate(sim.particles):
            Y[k, 6 * i:6 * i + 6] = (p.x, p.y, p.z, p.vx, p.vy, p.vz)
    if cache:
        np.savez(fname, t=t, Y=Y)
    return t, Y
