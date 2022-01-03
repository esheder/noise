"""Tools for Karhunen-Loeve expansion of the SDE model.

"""

import numpy as np
import scipy.linalg as linalg

from .parameters import Parameters


def _phi(k: int, m: np.ndarray, t: float, tot_t: float) -> np.ndarray:
    eigk = np.pi*(k-0.5) / tot_t
    denominator = m @ m + eigk ** 2
    numerator = m @ linalg.expm(m * t) - m * np.cos(eigk * t) + eigk * np.sin(eigk * t)
    return linalg.inv(denominator) @ numerator


def signal_make(ts: np.ndarray, par: Parameters, *,
                rand_gen: np.random.Generator,
                kl_terms: int,
                ) -> np.ndarray:
    print(len(ts))
    det_m = np.array(((par.α, 0), (par.λd, 0)), dtype=np.float64)
    sto_m = np.array(((par.σ1, -par.σ2), (0, par.σ2)), dtype=np.float64)
    tot_t = np.max(ts)
    noise = rand_gen.normal(0., 1., (2, kl_terms))
    kl_sol = np.array([np.sqrt(2/tot_t)
                       * sum(_phi(k+1, det_m, t, tot_t) @ (sto_m @ noise[:, k])
                             for k in range(kl_terms))[1]
                       for t in ts[1:]])
    y = np.zeros_like(ts, dtype=np.float64)
    y[1:] = par.equilibrium_flux*par.λd*np.diff(ts) + kl_sol
    return y
