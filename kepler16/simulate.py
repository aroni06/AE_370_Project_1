"""Convenience wrapper: build an initial state and march it with RK4."""

from . import constants as c
from .dynamics import rhs
from .orbits import initial_state
from .rk4 import integrate


def run(m_p, velocity="fixed", h=c.H_CHOSEN, t_final=c.T_FINAL, save_every=1):
    """Integrate one mass case. Returns (t, Y, masses)."""
    m, y0 = initial_state(m_p, velocity)
    t, Y = integrate(rhs, y0, t_final, h, args=(m,), save_every=save_every)
    return t, Y, m
