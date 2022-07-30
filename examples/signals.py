#!/usr/bin/env python
import numpy as np

from noise import Parameters
from noise.sdesolve import signal_make

#First, create a Parameters object. 
#In this case we create a -5000 pcm subcritical system with a generation time of 50 micro-seconds, 
# a source with a yield of 1 million neutrons per second, 
# and where the multiplicity distribution has an average of 2.42 and a second moment of 6. 
# The detection rate is set at 1e-4 detections per second on average.
par = Parameters.from_dubi(-500e-5, 5e-5, 1e6, 2.42, 6., 1e-4)
#Second, create a random number generator. The seed value of 48 here can be set to any other integer as well.
generator = np.random.Generator(np.random.SFC64(48))
#Decide on the experiment times. Let's say every 1ms for 10 minutes:
ts = np.linspace(0, 600., 600*1000)
#Generate a signal using the signal_make function
detections = signal_make(ts, par, rand_gen=generator)
#Congratulations, you have a detection signal. Live long and prosper.

