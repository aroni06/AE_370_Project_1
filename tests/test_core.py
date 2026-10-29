import numpy as np
import pytest

from kepler16 import constants as c
from kepler16.dynamics import accelerations, angular_momentum, energy, rhs, unpack
from kepler16.orbits import initial_state, relative_state, solve_kepler
from kepler16.rk4 import integrate, rk4_step


def test_initial_state_matches_table_2():
    # Table 2 of the methods draft, observed-mass baseline
    table2 = np.array([
        [0.000069776, 0.000350157, -0.059207448, 0.006646158, -0.000000848, 0.000144652],
        [0.000086119, -0.001191699, 0.200542080, -0.022601811, 0.000002883, -0.000483870],
        [-0.206223109, -0.000391851, 0.678035617, -0.018425189, 0.000002144, -0.005531925],
    ])
    _, y0 = initial_state(c.M_P0)
    assert np.allclose(y0.reshape(3, 6), table2, atol=1e-9)


@pytest.mark.parametrize("velocity", ["fixed", "rebuilt"])
@pytest.mark.parametrize("factor", [0.0, 0.5, 1.0, 2.0])
def test_barycenter_at_rest_at_origin(velocity, factor):
    m, y0 = initial_state(factor * c.M_P0, velocity)
    r, v = unpack(y0)
    assert np.allclose(m @ r, 0, atol=1e-15)
    assert np.allclose(m @ v, 0, atol=1e-17)


def test_relative_state_is_preserved_when_mass_changes():
    from kepler16.diagnostics import binary_separation, planet_relative
    m0, y0 = initial_state(c.M_P0, "fixed")
    m1, y1 = initial_state(c.M_P0 / 2, "fixed")
    for f in (binary_separation, planet_relative):
        for a, b in zip(f(y0, m0) if f is planet_relative else f(y0),
                        f(y1, m1) if f is planet_relative else f(y1)):
            assert np.allclose(a, b, atol=1e-15)


def test_forces_obey_newtons_third_law():
    m, y0 = initial_state(c.M_P0)
    r, _ = unpack(y0)
    assert np.allclose(m @ accelerations(r, m), 0, atol=1e-18)


def test_kepler_solver():
    M = np.linspace(0, 2 * np.pi, 50)
    for e in (0.0, 0.16, 0.9):
        E = solve_kepler(M, e)
        assert np.allclose(E - e * np.sin(E), M, atol=1e-13)


def test_analytic_orbit_conserves_two_body_energy():
    mu = c.G * c.M_STARS
    r, v = relative_state(c.BINARY, mu, np.linspace(0, 100, 37))
    eps = 0.5 * np.sum(v * v, axis=1) - mu / np.linalg.norm(r, axis=1)
    assert np.allclose(eps, -mu / (2 * c.BINARY["a"]), rtol=1e-12)


def test_rk4_amplification_factor():
    lam, h = -0.7 + 1.3j, 0.3
    y1 = rk4_step(lambda t, y: lam * y, 0.0, np.array([1.0 + 0j]), h)
    z = h * lam
    assert np.isclose(y1[0], 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24)


def test_rk4_fourth_order_on_scalar_ode():
    errs = []
    for h in (0.1, 0.05):
        t, Y = integrate(lambda t, y: np.array([np.cos(t) * y[0]]), [1.0], 2.0, h)
        errs.append(abs(Y[-1, 0] - np.exp(np.sin(2.0))))
    assert 3.8 < np.log2(errs[0] / errs[1]) < 4.2


def test_short_run_conserves_energy_and_momentum():
    m, y0 = initial_state(c.M_P0)
    t, Y = integrate(rhs, y0, 41.0, 0.1, args=(m,))
    E = energy(Y, m)
    L = angular_momentum(Y, m)
    assert np.max(np.abs(E / E[0] - 1)) < 1e-8
    assert np.max(np.linalg.norm(L - L[0], axis=1)) / np.linalg.norm(L[0]) < 1e-8
    r, v = unpack(Y)
    assert np.allclose(np.einsum("i,tij->tj", m, v), 0, atol=1e-16)
