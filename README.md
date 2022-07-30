# Noise Experiment Simulation Using SDEs
Feynman-Y based noise experiments can measure the α eigenvalue of the reactor. 
This is done using the variance to mean ratio of the number of detections for a given time window size while listening to the reactor noise.
Chen Dubi et al. have developed a coupled 2 SDE system that has the same variance and mean as the branching process itself.
Using that model we can generate for any core parameters a set of detection signals by solving an SDE.
This should be typically much faster than the existing methods that are based on Monte Carlo simulations of the time based branching process.

## How to use this code
We mostly expect researchers to want one of two things: Either they want to generate experiment-like signals of detections in a given system, or they want to skip the signal itself and send the results through a Feynman-Y fitting procedure to retrieve results that simulate those from an experiment.

To do this, we supply two main interfaces, as explained below. Examples are given in the examples folder in both a Jupyter notebook and a Python script format.
The code under `noise.analog` is used for verification of our solution, and has a similar structure, so once you understand the two interfaces below, using the analog solver should be fairly trivial for you as well.

### Generating a signal
To generate an SDE signal, the following steps are necessary:
1. Create a Parameters object. 
	This data object holds the physical information about the system used by the SDE solver. 
	There are 3 methods for generating this object. The direct way is to just write `Parameters(alpha, source, sigma1, sigma2, detection_rate)`, and there are two class methods as well that allow a proper definition using other system parameters such as the various reaction rates (the `from_xs` classmethod), or system reactivity (the `from_dubi` classmethod).
2. Create a random number generator. 
	Imporing NumPy as np and then using `np.random.Generator(np.random.SFC64(seed))` will create a random number generator with the given seed that should work well.
	Researchers can use other random number generators if they want.
3. Call the `signal_make` function from `noise.sdesolve`. 
	This function takes an array of times at which the detections should be calculated, the parameters of the system, and the random number generator.
	Researchers may also provide their own implementation for the SDE solver if they so desire, and three Numba based solvers are available for Forward Euler, Backward Euler and a first order Theta-method Euler implementation.
	The default solver is the Forward Euler solver.
	Calling this function returns an array of detections at the desired times.

### Getting a Feynman-Y fit
To get the results of a Feynman-Y fit to a simulated experiment, one can use the follwing process:
1. Create a Parameters object, see step 1 above
2. Create a random number generator, see step 2 above
3. Call `feynman_y_by_signal` function under `noise.sdesolve`.
	This function accepts the system parameters (see step 1 above), how long the experiment is, a strategy for generating a signal (see the `signal_make` method in the signal generation process, which is set by default), and a random number generator (see step 2 above).
	The function generates a signal, bins it in different window sizes, and returns a triplet of arrays. The first is for the window sizes, the second is for the Feynman-Y Variance-to-Mean-Minus-1 ratio, and the third is inversely proportional to the assumed statistical noise, due to there being less non-overlapping windows of larger sizes in a given experiment length.
4. Filter the data to a range of choise.
	We usually do not consider window sizes larger than 0.1s when doing our Feynman-Y fits because our model does not handle delayed neutrons which would have a significant effect for windows of this size.
5. Fit the window size, Feynman-Y values and assumed errors to the Feynman-Y curve using the `fit_to_feynman_y` function from `noise.analytic`. 
	This function takes the triplet of arrays (the assumed statistical noise estimate may be omitted).
	It returns a fit result from `scipy.optimize.curve_fit` for the Feynman-Y curve, which is a pair of fit parameters and fit covariances.
	Researchers that fit to a different curve may of course use their own curve fitting tools.
	
## Project Structure
Under the `noise` directory you will find our package. 
For most applications, researchers will just need this directory and the `setup.py` file to get things started.

Under the `examples` directory you will find examples for how to use the code. These appear as both Python scripts and Jupyter notebooks.

Under the `hpc_runs` directory you will find the data analysis used in our upcoming paper, as well as scripts we used to generate said data.
The `data_analysis.ipynb` file in this directory is directly used to generate our figures and estimates.
The `analysis_data.gz` file is a compressed `.tar` file that includes the csv files used in this analysis, and can be opened with many decompression software.
The Python and Bash files in that directory are the scripts we used to generate the csv files on the NegevHPC cluster. They are there just in case we want to reproduce our results on that cluster in the future, and are not meant for use by other researchers. You may view them as additional examples of lower quality, if you like.

Under the `tests` directory you will find our unit tests.
We could have more of these, and if we get more researchers interested in this code this would be higher own our to-do list.
You can consider contributing tests yourself, actually. It's a good place for new contributers to start.

## Implementation
Our implementation is currently done in Python. This could change later if found to be necessary.

