"""Classical fourth-order Runge-Kutta time marcher, written from scratch."""

import numpy as np


def rk4_step(f, t, y, h, *args):
    """Advance y' = f(t, y, *args) from t to t + h with one RK4 step (Eqs. 13-17)."""
    k1 = f(t, y, *args)
    k2 = f(t + h / 2, y + h / 2 * k1, *args)
    k3 = f(t + h / 2, y + h / 2 * k2, *args)
    k4 = f(t + h, y + h * k3, *args)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(f, y0, t_final, h, args=(), save_every=1):
    """March from t = 0 to t_final with fixed step h.

    t_final / h must be a whole number of steps, so every run lands exactly on
    t_final. The state is stored every `save_every` steps, so runs with
    different h can share output times when save_every * h is the same.

    Returns (t, Y) with t of shape (n_out,) and Y of shape (n_out, len(y0)).
    """
    n_steps = int(round(t_final / h))
    if abs(n_steps * h - t_final) > 1e-9 * max(1.0, t_final):
        raise ValueError(f"t_final={t_final} is not a multiple of h={h}")
    if n_steps % save_every:
        raise ValueError("number of steps must be a multiple of save_every")

    n_out = n_steps // save_every + 1
    Y = np.empty((n_out, len(y0)))
    Y[0] = y0
    y = np.array(y0, dtype=float)
    for k in range(1, n_steps + 1):
        y = rk4_step(f, (k - 1) * h, y, h, *args)
        if k % save_every == 0:
            Y[k // save_every] = y
    t = np.arange(n_out) * save_every * h
    return t, Y
