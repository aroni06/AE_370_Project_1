"""Physical constants and published Kepler-16 parameters (Doyle et al. 2011).

Units: astronomical units (AU), days, and solar masses throughout.
"""

# Gaussian gravitational constant squared, AU^3 / (Msun day^2)
G = 0.01720209895**2

# Jupiter mass in solar masses (IAU nominal GM ratio)
MJUP_MSUN = 9.547919e-4

M_A = 0.6897          # Kepler-16 A
M_B = 0.20255         # Kepler-16 B
M_P0 = 0.333 * MJUP_MSUN   # Kepler-16 b, nominal
M_STARS = M_A + M_B

# Orbital elements. Angles in degrees; "lam" is the mean longitude at t = 0.
BINARY = dict(a=0.22431, e=0.15944, inc=90.3401, Omega=0.0,
              omega=263.464, lam=92.352, P=41.079220)
PLANET = dict(a=0.7048, e=0.0069, inc=90.0322, Omega=0.003,
              omega=318.0, lam=106.51, P=228.776)

# Default study settings
T_FINAL = 4576.0              # ~20 nominal planetary periods (20 x 228.776 = 4575.5 d), rounded so every tested h divides it
H_CHOSEN = 0.1                # time step chosen by the time-step study (days)
