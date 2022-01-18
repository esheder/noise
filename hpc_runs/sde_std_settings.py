#!/usr/bin/env python
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from noise.parameters import Parameters
from noise.analytic import fit_to_feynman_y
from noise.sdesolve import feynman_y_by_signal

multiplicity = np.array([0.13, 0.22, 0.2, 0.2, 0.1, 0.1, 0.05])
nu = (multiplicity * np.arange(len(multiplicity))).sum()
moment2 = (multiplicity * (np.arange(len(multiplicity)) ** 2)).sum()
lifetimes = np.logspace(-6, -4, num=10)
sources = np.logspace(3, 7, num=10)
detections = np.logspace(-6, -2, num=10)
times = np.logspace(np.log10(60), np.log10(10*3600), num=10)
parameters = ('Reactivity', 'Lifetime', 'Source', 'Detection Rate', 'Measurement Time', 'Seed', 'Inf', 'Alpha',
              'CovInf', 'CovInfAlpha', 'CovAlphaInf', 'ConvAlphaAlpha')
defaults = {'Λ': lifetimes[-2], 's': sources[5], 'pd': detections[5]}
vectors = {'Λ': lifetimes, 's': sources, 'pd': detections}
arguments = ['s', 'pd'] + ['time']


if __name__ == '__main__':
    parser = ArgumentParser(description='Tool for creating signals')
    parser.add_argument('index', type=int, help="Index in the options for the argument to use")
    parser.add_argument('reactivity', type=float, help="The system reactivity to test at")
    parser.add_argument('--pdir', type=Path, help="Path to dir to use",
                        default=Path.cwd())
    parser.add_argument('--seeds',
                        type=int,
                        help="Maximal seed to use in generator",
                        default=1000)
    parser.add_argument('--batch', type=int, help="Batch of calculations to run.", default=50)
    parser.add_argument('--total', action='store_true', help="Flag to say how big index should go")
    parser.add_argument('--arguments', action='store_true', help="Flag to print out the argument names")
    args = parser.parse_args()
    if args.total:
        print(int(len(arguments)*10*args.seeds/args.batch))
        exit(0)
    elif args.arguments:
        print(','.join(parameters))
        exit(0)
    rundex = args.index*args.batch
    superdex, seed0 = divmod(rundex-args.batch, args.seeds)
    index, argdex = divmod(superdex, len(arguments))
    argument = arguments[argdex]
    print(f'{argument=},{index=},{seed0=}')
    if argument in defaults:
        defaults[argument] = vectors[argument][index]
        t = times[0]
    elif argument == 'time':
        t = times[index]
    else:
        raise ValueError("Argument must be of a given set: "
                         f"{arguments}")
    par = Parameters.from_dubi(ν=nu, ν2=moment2, ρ=-args.reactivity, **defaults)
    res = {}
    seeds = range(seed0, seed0+args.batch+1)
    for seed in seeds:
        generator = np.random.Generator(np.random.SFC64(seed))
        ts, curve, sigma = feynman_y_by_signal(par, t, rand_gen=generator)
        mask = (ts <= 1e-1)
        tfit, cfit, sfit = ts[mask], curve[mask], sigma[mask]
        popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
        res[seed] = (popt, pcov)
    with (args.pdir / f'sde_std.reac{args.reactivity:.5e}.{args.index}.csv').open('w') as f:
        for seed in seeds:
            popt, pcov = res[seed]
            f.write(f'{",".join([f"{x:.6e}" for x in defaults.values()])},{-args.reactivity},'
                    f'{t:.6e},{seed},'
                    f'{popt[0]:.5e},{popt[1]:.5e},'
                    f'{pcov[0, 0]:.5e},{pcov[0, 1]:.5e},'
                    f'{pcov[1, 0]:.5e},{pcov[1, 1]:.5e}\n')
