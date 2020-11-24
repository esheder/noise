# Noise Experiment Simulation Using SDEs
Feynman-Y based noise experiments can measure the α eigenvalue of the reactor. 
This is done using the variance to mean ratio of the number of detections for a given time window size while listening to the reactor noise.
Chen Dubi et al. have developed a coupled 2 SDE system that has the same variance and mean as the branching process itself.
Using that model we can generate for any core parameters a set of detection signals by solving an SDE.
This should be typically much faster than the existing methods that are based on Monte Carlo simulations of the time based branching process.

## Implementation
Our implementation is currently done in Python. This may change later to improve speed, if necessary.

## To Do List
1. Implement an analog solution.
1. Implement the methods currently used elsewhere.
1. Compare relative costs.
1. Write a paper.
