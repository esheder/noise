#!/usr/bin/env python
from argparse import ArgumentParser

import numpy as np

from noise import Parameters
from noise.analytic import fit_to_feynman_y
from noise.sdesolve import feynman_y_by_signal

multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])


if __name__ == '__main__':
    parser = ArgumentParser(description='Tool for creating signals')
    parser.add_argument('seed', type=int, help="Seed to use for the random generator")
    args = parser.parse_args()
    generator = np.random.Generator(np.random.SFC64(args.seed))
    nu = (multiplicity * np.arange(len(multiplicity))).sum()
    moment2 = (multiplicity * (np.arange(len(multiplicity)) ** 2)).sum()
    par = Parameters.from_dubi(-500e-5, 5e-5, 1e6, nu, moment2, 1e-4)
    t = 600.
    ts, curve, sigma = feynman_y_by_signal(par, t, rand_gen=generator)
    mask = (ts <= 1e-1)
    tfit, cfit, sfit = ts[mask], curve[mask], sigma[mask]
    popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
    with open(f'seed.{args.seed}', 'w') as f:
        f.write(f'{args.seed},{popt[0]:.5e},{popt[1]:.5e}\n')
