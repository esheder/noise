"""We have a stochastic process. Neutrons are born in the core randomly. Sometimes they die, sometimes they cause
fission and rarely they are observed and die. We want to have a signal of counts within time windows, and we want
to observe the mean and variance of said counts in said windows.

"""
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from itertools import chain
from typing import Generator, Sequence, Tuple
import logging

import numpy as np


logger = logging.getLogger(__name__)


class Process(IntEnum):
    Detection = 0
    Fission = 1
    Loss = 2


@dataclass
class AnalogParameters:
    """Parameters used in the Analog model.

    Parameters
    ----------
    lifetime - Neutron lifetime, in seconds.
    s - Source rate, in 1/sec.
    pf - Fission probability.
    pd - Detection probability.
    multiplicity - Multiplicity probability vector.
    tmax - Time when the experiment ends.

    """
    lifetime: float
    s: float
    pf: float
    pd: float
    multiplicity: Sequence[float]
    tmax: float

    @property
    def prob_vector(self) -> Tuple[float, ...]:
        return self.pd, self.pf, 1. - self.pd - self.pf

    @property
    def k(self) -> float:
        return float(self.pf*np.sum(np.arange(len(self.multiplicity))*self.multiplicity))

    @property
    def ρ(self) -> float:
        return 1. - (1. / self.k)

    @property
    def α(self) -> float:
        return self.ρ / self.lifetime


def spread_sources(t: float, s: float, randgen: np.random.Generator) -> np.array:
    points = int(s * t)
    dt = randgen.exponential(scale=1./s, size=(points,))
    ts = np.cumsum(dt)
    return ts[ts < t]


def _detections(sources: np.array, par: AnalogParameters, randgen: np.random.Generator
                ) -> Generator[float, None, None]:
    initial = len(sources)
    so_far = 0
    estimated = 0.5 * initial / (1. - par.k)
    last = 0.
    while len(sources):
        size = (len(sources), )
        lags = randgen.exponential(par.lifetime, size=size)
        tevents = sources + lags
        processes = randgen.choice(a=[e.value for e in Process], p=par.prob_vector, size=size)
        yield from (t for t, process in zip(tevents, processes)
                    if process == Process.Detection and t <= par.tmax)
        fission_times = tuple(t for t, process in zip(tevents, processes)
                              if process == Process.Fission and t <= par.tmax)
        new_size = (len(fission_times),)
        multiples = randgen.choice(a=np.arange(len(par.multiplicity)), p=par.multiplicity, size=new_size)
        topdex = multiples.cumsum()
        sources = np.zeros(new_size, dtype=np.float)
        if new_size[0]:
            sources[:topdex[0]] = fission_times[0]
            for k, (i, j) in enumerate(zip(topdex[:-1], topdex[1:])):
                sources[i:j] = fission_times[k+1]
        so_far += size[0]
        covered = so_far/estimated
        if covered > last + 0.1:
            logger.info(f'Finished {covered:3.0%}')
            last = covered


def detections(*args, **kwargs) -> np.array:
    return np.fromiter(_detections(*args, **kwargs), dtype=np.float)


def _feynman_y(signal: np.array, tmax: float) -> Generator[Tuple[float, float, float], None, None]:
    dts = np.logspace(-3, np.log(tmax)/2, 30)
    for dt in dts:
        bins = np.linspace(0., tmax, int(tmax//dt))
        hist, edges = np.histogram(signal, bins=bins)
        if len(hist) > 1:
            fy = (hist.var(ddof=1) / hist.mean()) - 1.
            yield edges[1], fy, 1.


def feynman_y(signal: np.array, tmax: float) -> Tuple[np.array, np.array, np.array]:
    ts, fy, sigma = map(np.array, zip(*list(_feynman_y(signal, tmax))))
    return ts, fy, sigma
