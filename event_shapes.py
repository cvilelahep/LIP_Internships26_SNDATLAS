"""
Event-shape (event-level) variables for classifying particle-collision events.

Input convention
-----------------
Each event is a numpy array of shape (N, 3): the 3-momenta (px, py, pz) of the
N particles in that event (in GeV, or any consistent unit).
For 4-vectors (E, px, py, pz), just pass momenta[:, 1:].

Provided:
  - sphericity_tensor_eigs
  - sphericity, aplanarity, planarity
  - thrust, thrust_axis
  - c_parameter, d_parameter
  - fox_wolfram_moments
  - transverse_sphericity  (2D analogue, common at hadron colliders)
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_legendre


# ---------------------------------------------------------------------------
# 1. Sphericity tensor -> sphericity / aplanarity / planarity
# ---------------------------------------------------------------------------
def sphericity_tensor_eigs(p, r=2.0):
    """
    Build the (possibly generalized) momentum tensor
        S^{ab} = sum_i |p_i|^(r-2) p_i^a p_i^b / sum_i |p_i|^r
    and return eigenvalues sorted descending (lambda1 >= lambda2 >= lambda3).
    r=2 gives the standard sphericity tensor.
    """
    p = np.asarray(p, dtype=float)
    mag = np.linalg.norm(p, axis=1)
    mag = np.where(mag == 0, 1e-12, mag)  # avoid div by zero
    weight = mag ** (r - 2)

    # weighted outer-product sum: S^{ab} = sum_i w_i p_i^a p_i^b
    S = np.einsum('i,ia,ib->ab', weight, p, p)
    S /= np.sum(mag ** r)

    eigvals = np.linalg.eigvalsh(S)          # ascending
    eigvals = eigvals[::-1]                   # descending: l1 >= l2 >= l3
    return eigvals  # sums to 1


def sphericity(p):
    l1, l2, l3 = sphericity_tensor_eigs(p)
    return 1.5 * (l2 + l3)


def aplanarity(p):
    l1, l2, l3 = sphericity_tensor_eigs(p)
    return 1.5 * l3


def planarity(p):
    l1, l2, l3 = sphericity_tensor_eigs(p)
    return l2 - l3


# ---------------------------------------------------------------------------
# 2. Thrust (infrared/collinear safe) — found by maximizing over unit axis
# ---------------------------------------------------------------------------
def thrust(p, n_restarts=12, seed=0):
    """
    T = max_n  sum_i |p_i . n| / sum_i |p_i|,  |n| = 1

    Non-convex optimization -> multi-start local optimizer over the sphere.
    For small N (<~ a dozen particles) you could brute-force via combinatorics
    instead; this numerical approach scales to any N.
    """
    p = np.asarray(p, dtype=float)
    denom = np.sum(np.linalg.norm(p, axis=1))
    if denom == 0:
        return 0.0, np.array([0.0, 0.0, 1.0])

    rng = np.random.default_rng(seed)

    def neg_thrust(angles):
        theta, phi = angles
        n = np.array([np.sin(theta) * np.cos(phi),
                      np.sin(theta) * np.sin(phi),
                      np.cos(theta)])
        return -np.sum(np.abs(p @ n)) / denom

    best_val = -1.0
    best_axis = None
    for _ in range(n_restarts):
        x0 = rng.uniform([0, 0], [np.pi, 2 * np.pi])
        res = minimize(neg_thrust, x0, method='Nelder-Mead')
        if -res.fun > best_val:
            best_val = -res.fun
            theta, phi = res.x
            best_axis = np.array([np.sin(theta) * np.cos(phi),
                                   np.sin(theta) * np.sin(phi),
                                   np.cos(theta)])

    # sign convention: axis direction doesn't matter for T, normalize sign
    if best_axis[2] < 0:
        best_axis = -best_axis
    return best_val, best_axis


def thrust_axis(p, **kwargs):
    _, n = thrust(p, **kwargs)
    return n


# ---------------------------------------------------------------------------
# 3. C-parameter and D-parameter (from linearized momentum tensor, r=1)
# ---------------------------------------------------------------------------
def c_parameter(p):
    l1, l2, l3 = sphericity_tensor_eigs(p, r=1.0)
    return 3.0 * (l1 * l2 + l2 * l3 + l3 * l1)


def d_parameter(p):
    l1, l2, l3 = sphericity_tensor_eigs(p, r=1.0)
    return 27.0 * l1 * l2 * l3


# ---------------------------------------------------------------------------
# 4. Fox-Wolfram moments
# ---------------------------------------------------------------------------
def fox_wolfram_moments(p, lmax=4):
    """
    H_l = sum_{i,j} |p_i||p_j| P_l(cos theta_ij) / E_vis^2

    Here we approximate E_vis by sum of |p_i| (massless approx); pass true
    energies via `energies` if available for exact E_vis normalization.
    """
    p = np.asarray(p, dtype=float)
    mag = np.linalg.norm(p, axis=1)
    E_vis = np.sum(mag)
    if E_vis == 0:
        return np.zeros(lmax + 1)

    # cos(theta_ij) matrix
    unit = p / mag[:, None]
    cos_ij = unit @ unit.T
    cos_ij = np.clip(cos_ij, -1.0, 1.0)

    mag_outer = np.outer(mag, mag)

    H = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        Pl = eval_legendre(l, cos_ij)
        H[l] = np.sum(mag_outer * Pl) / E_vis ** 2
    return H  # H[0] should be ~1


def fox_wolfram_R2(p):
    H = fox_wolfram_moments(p, lmax=2)
    return H[2] / H[0]


# ---------------------------------------------------------------------------
# 5. Transverse sphericity (2D, common at hadron colliders like LHC)
# ---------------------------------------------------------------------------
def transverse_sphericity(p):
    """
    Same idea as sphericity but using only (px, py) components -- appropriate
    at hadron colliders where the boost along the beam (z) is not physical
    for shape purposes.
    S_T = 2 * lambda2_T / (lambda1_T + lambda2_T)
    """
    pt = np.asarray(p, dtype=float)[:, :2]
    mag2 = np.sum(pt ** 2, axis=1)
    S = np.einsum('i,ia,ib->ab', np.ones_like(mag2), pt, pt)
    S /= np.sum(mag2)
    eigvals = np.linalg.eigvalsh(S)[::-1]  # l1 >= l2
    l1, l2 = eigvals
    return 2.0 * l2 / (l1 + l2)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def make_dijet_event(n_per_jet=15, jet_spread=0.05):
        """Two back-to-back jets along +/- z, each a cluster of particles."""
        axis = np.array([0, 0, 1.0])
        parts = []
        for sign in (+1, -1):
            base = sign * axis * 20.0
            jet = base + rng.normal(scale=jet_spread * 20.0, size=(n_per_jet, 3))
            parts.append(jet)
        return np.vstack(parts)

    def make_isotropic_event(n=30, p_scale=10.0):
        """Isotropic 'spherical' event: random directions, similar magnitude."""
        dirs = rng.normal(size=(n, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        mags = p_scale * (0.7 + 0.6 * rng.random(n))
        return dirs * mags[:, None]

    for name, ev in [("dijet", make_dijet_event()), ("isotropic", make_isotropic_event())]:
        S = sphericity(ev)
        A = aplanarity(ev)
        T, axis = thrust(ev)
        C = c_parameter(ev)
        D = d_parameter(ev)
        H = fox_wolfram_moments(ev, lmax=2)
        ST = transverse_sphericity(ev)

        print(f"--- {name} event ---")
        print(f"  sphericity          S  = {S:.3f}")
        print(f"  aplanarity          A  = {A:.3f}")
        print(f"  thrust              T  = {T:.3f}")
        print(f"  C-parameter         C  = {C:.3f}")
        print(f"  D-parameter         D  = {D:.3f}")
        print(f"  Fox-Wolfram R2      R2 = {H[2]/H[0]:.3f}")
        print(f"  transverse spher.   ST = {ST:.3f}")
        print()
