#!/usr/bin/env python
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from time import time

import numpy as np

from noise.parameters import Parameters
from noise.analytic import fit_to_feynman_y
from noise.sdesolve import feynman_y_by_signal

multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])
nu = (multiplicity * np.arange(len(multiplicity))).sum()
moment2 = (multiplicity * (np.arange(len(multiplicity)) ** 2)).sum()
reactivities = -1e-5*np.logspace(np.log10(50), 3, num=10)
lifetimes = np.logspace(-6, -4, num=10)
sources = np.logspace(3, 7, num=10)
detections = np.logspace(-6, -2, num=10)
times = np.logspace(np.log10(60), np.log10(3600), num=10)
defaults = {'ρ': reactivities[-1], 'Λ': lifetimes[-2],
            's': sources[5], 'pd': detections[5]}
vectors = {'ρ': reactivities, 'Λ': lifetimes,
           's': sources, 'pd': detections}
arguments = list(vectors.keys()) + ['time']


def time_f(f):
    @wraps(f)
    def _wraps(*args, **kwargs):
        start = time()
        res = f(*args, **kwargs)
        end = time()
        return end-start, res
    return _wraps


if __name__ == '__main__':
    parser = ArgumentParser(description='Tool for creating signals')
    parser.add_argument('index', type=int, help="Index in the options for the argument to use")
    parser.add_argument('--pdir', type=Path, help="Path to dir to use",
                        default=Path.cwd())
    parser.add_argument('--seeds',
                        type=int,
                        help="Maximal seed to use in generator",
                        default=50)
    args = parser.parse_args()
    superdex, seed = divmod(args.index-1, args.seeds)
    index, argdex = divmod(superdex, len(arguments))
    argument = arguments[argdex]
    generator = np.random.Generator(np.random.SFC64(seed))
    print(f'{argument=},{index=},{seed=}')
    if argument in defaults:
        defaults[argument] = vectors[argument][index]
        t = times[0]
    elif argument == 'time':
        t = times[index]
    else:
        raise ValueError("Argument must be of a given set: "
                         f"{arguments}")
    par = Parameters.from_dubi(ν=nu, ν2=moment2, **defaults)
    elapsed, (ts, curve, sigma) = time_f(feynman_y_by_signal)(par, t, rand_gen=generator)
    mask = (ts <= 1e-1)
    tfit, cfit, sfit = ts[mask], curve[mask], sigma[mask]
    popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
    with (args.pdir / f'sde_timing.{argument}.{index}.{seed}').open('w') as f:
        f.write(f'{",".join([f"{x:.6e}" for x in defaults.values()])},'
                f'{t:.6e},{seed},{elapsed:.4e},'
                f'{popt[0]:.5e},{popt[1]:.5e}\n')
