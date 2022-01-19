#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path
import itertools as it

import numpy as np

from noise.parameters import Parameters
from noise.analytic import fit_to_feynman_y
from noise.sdesolve import feynman_y_by_signal

multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])
nu = (multiplicity * np.arange(len(multiplicity))).sum()
moment2 = (multiplicity * (np.arange(len(multiplicity)) ** 2)).sum()
reactivities = -np.logspace(np.log10(5e-4), -1, 10)
sources = np.logspace(3, 7, num=20)
detections = np.logspace(-6, -2, num=20)
time = 600  # 10 minutes
lifetime = 50e-6  # Thermal reactor
parameters = ('Reactivity', 'Lifetime', 'Source', 'Detection Rate', 'Measurement Time', 'Seed', 'Inf', 'Alpha',
              'CovInf', 'CovInfAlpha', 'CovAlphaInf', 'ConvAlphaAlpha')
combinations = tuple(it.product(reactivities, sources, detections))


def choose_parameters(index: int, batch: int, seeds: int):
    rundex = index*batch
    superdex, seed0 = divmod(rundex-batch, seeds)
    _, argdex = divmod(superdex, len(combinations))
    r, s, d = combinations[argdex]
    return seed0, r, s, d


def main():
    parser = ArgumentParser(description='Tool for creating signals')
    parser.add_argument('index', type=int, help="Index in the options for the argument to use")
    parser.add_argument('--pdir', type=Path, help="Path to dir to use",
                        default=Path.cwd())
    parser.add_argument('--seeds',
                        type=int,
                        help="Maximal seed to use in generator",
                        default=1000)
    parser.add_argument('--batch', type=int, help="Batch of calculations to run.", default=500)
    parser.add_argument('--total', action='store_true', help="Flag to say how big index should go")
    parser.add_argument('--arguments', action='store_true', help="Flag to print out the argument names")
    parser.add_argument('--test', action='store_true', help="Flag to print out the possible parameter values")
    args = parser.parse_args()
    total = len(combinations) * args.seeds // args.batch
    if args.total:
        print(total)
        exit(0)
    elif args.arguments:
        print(','.join(parameters))
        exit(0)
    if args.seeds % args.batch != 0:
        raise ValueError("Seed size must be a multiple of the batch size")

    if args.test:
        for i in range(1, total + 1):
            seed0, reactivity, source, pd = choose_parameters(i, args.batch, args.seeds)
            print(f'{reactivity=},{source=},{pd=}')
        exit(0)

    seed0, reactivity, source, pd = choose_parameters(args.index, args.batch, args.seeds)
    print(f'{reactivity=},{source=},{pd=}')

    par = Parameters.from_dubi(ν=nu, ν2=moment2, ρ=reactivity, s=source, Λ=lifetime, pd=pd)
    res = {}
    seeds = range(seed0, seed0 + args.batch + 1)
    for seed in seeds:
        generator = np.random.Generator(np.random.SFC64(seed))
        ts, curve, sigma = feynman_y_by_signal(par, time, rand_gen=generator)
        mask = (ts <= 1e-1)  # To avoid the need for delayed neutrons
        tfit, cfit, sfit = ts[mask], curve[mask], sigma[mask]
        popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
        res[seed] = (popt, pcov)

    with (args.pdir / f'sde_std.{args.index}.csv').open('w') as f:
        for seed in seeds:
            popt, pcov = res[seed]
            f.write(f'{reactivity:.5e},{lifetime:.5e},{source:.5e},{pd:.5e},{time:.5e},'
                    f'{seed},'
                    f'{popt[0]:.5e},{popt[1]:.5e},'
                    f'{pcov[0, 0]:.5e},{pcov[0, 1]:.5e},'
                    f'{pcov[1, 0]:.5e},{pcov[1, 1]:.5e}\n')


if __name__ == '__main__':
    main()
