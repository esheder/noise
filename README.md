# Noise Experiment Simulation Using SDEs
Feynman-Y based noise experiments can measure the α eigenvalue of the reactor. 
This is done using the variance to mean ratio of the number of detections for a given time window size while listening to the reactor noise.
Chen Dubi et al. have developed a coupled 2 SDE system that has the same variance and mean as the branching process itself.
Using that model we can generate for any core parameters a set of detection signals by solving an SDE.
This should be typically much faster than the existing methods that are based on Monte Carlo simulations of the time based branching process.

## How to use this code
We mostly expect researchers to want one of two things: Either they want to generate experiment-like signals of detections in a given system, or they want to skip the signal itself and send the results through a Feynman-Y fitting procedure to retrieve results that simulate those from an experiment.

To do this, we supply two main interfaces, as explained below. Examples are given in the [examples](https://github.com/esheder/noise/tree/master/examples) folder in both a Jupyter notebook and a Python script format.
We recommend looking over there first.
The code under `noise.analog` is used for verification of our solution, and has a similar structure, so once you understand the two interfaces below, using the analog solver should be fairly trivial for you as well.

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

