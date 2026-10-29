# AE 370 Project 1: Halving the mass of Kepler-16 b

How does reducing the mass of Kepler-16 b by 50% affect the orbital dynamics of
the Kepler-16 three-body system? This repository integrates the Newtonian
three-body problem with a classical fourth-order Runge–Kutta (RK4) method
written from scratch, verifies it, and runs the mass experiment.

## Layout

```
kepler16/            the numerical method and model
  constants.py       G, masses, published orbital elements (Doyle et al. 2011)
  orbits.py          Kepler's equation, elements -> Cartesian state, initial states
  dynamics.py        f(t, y) for the 18-component state, energy, angular momentum
  rk4.py             fixed-step RK4 time marcher (rk4_step, integrate)
  diagnostics.py     planet offset D_p, in-plane phase, revolution and conjunction times
  reference.py       REBOUND IAS15 reference solutions (verification only)
  simulate.py        build an initial state and run RK4
studies/             one script per section of the report
  verify_two_body.py       RK4 vs exact Kepler orbit (m_p = 0)
  verify_conservation.py   energy, angular momentum, momentum vs h
  verify_rebound.py        RK4 vs IAS15, full three-body problem
  timestep_study.py        error in each diagnostic vs h, compared with effect size
  mass_experiment.py       cases A, B, C and the mass sweep
tests/               pytest unit tests (initial state vs Table 2, RK4 order, ...)
figures/             generated figures (PDF and PNG)
results/             generated numbers (JSON)
run_all.py           runs every study in report order
```

The time integrator never calls an outside solver. REBOUND appears only in
`kepler16/reference.py`, to produce independent benchmark solutions.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # unit tests, under a second
python run_all.py           # all studies and figures, about 3 minutes
python -m studies.verify_two_body   # or any single study
```

## Cases

| Case | Planet mass | Planet's initial state relative to the stars |
| --- | --- | --- |
| A | m_p0 = 0.333 M_Jup | built from Table 1 elements with mu = G(M* + m_p0) |
| B | m_p0 / 2 | same relative position and velocity as A |
| C | m_p0 / 2 | rebuilt from the same elements with mu = G(M* + m_p0/2) |

The sweep repeats B and C for m_p / m_p0 in {0, 0.25, 0.5, 1, 2}.

## Figures

| Figure | Script |
| --- | --- |
| `verify_two_body` | `studies/verify_two_body.py` |
| `verify_conservation` | `studies/verify_conservation.py` |
| `verify_rebound` | `studies/verify_rebound.py` |
| `timestep_study` | `studies/timestep_study.py` |
| `results_planet_displacement`, `results_conjunction_shift`, `results_mass_sweep` | `studies/mass_experiment.py` |

## Units

AU, days and solar masses, with G = k^2, k = 0.01720209895.
