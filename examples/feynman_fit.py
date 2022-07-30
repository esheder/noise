#!/usr/bin/env python
import numpy as np

from noise import Parameters
from noise.sdesolve import feynman_y_by_signal
from noise.analytic import fit_to_feynman_y

#First, create a Parameters object. 
#In this case we create a -5000 pcm subcritical system with a generation time of 50 micro-seconds, 
# a source with a yield of 1 million neutrons per second, 
# and where the multiplicity distribution has an average of 2.42 and a second moment of 6. 
# The detection rate is set at 1e-4 detections per second on average.
par = Parameters.from_dubi(-500e-5, 5e-5, 1e6, 2.42, 6., 1e-4)
#Second, create a random number generator. The seed value of 48 here can be set to any other integer as well.
generator = np.random.Generator(np.random.SFC64(48))
#Decide on the experiment length. Let's say 10 minutes:
t = 600.
#Generate a Feynman-Y curve
window_sizes, fy, sigma = feynman_y_by_signal(par, t, rand_gen=generator)
#Filter down the results to 0.1s
mask = (window_sizes <= 1e-1)
tfit, cfit, sfit = window_sizes[mask], fy[mask], sigma[mask]
#Fit the results to a Feynman-Y curve
popt, pcov = fit_to_feynman_y(tfit, cfit, sfit)
#Congratulations, you have the fitted values. For example, alpha is just popt[1]:
print(popt[1])

