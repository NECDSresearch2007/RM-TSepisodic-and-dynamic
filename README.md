# Project 
This repository provides experimental code for our paper  "Learning with Posterior sampling for Revenue Management under Time-varying Demand" accepted at the 33rd International Joint Conference on Artificial Intelligence (IJCAI 2024). 

# Requirements
We conducted the numerical experiments with the following requirements: 
```
'cvxopt': '1.2.6',
 'GPy': '1.10.0', 
 'matplotlib': '3.5.1',
 'numpy': '1.20.3',
 'Python': '3.8.13', 
 'Gurobi': '10.0.0'.
```
Note that we obtained the experiments in the paper based on Gurobi optimizer, which is a commercial solver (https://www.gurobi.com/), 
but the almost same results can be obtained with cvxopt, which is a free solver (https://cvxopt.org/), instead. 
Here, we provide an option to choose the two solvers in the experimental setting file. 

# Usage
## To run experiments
Change to the directory and run experiments for algorithms, "__TS-episodic__", "__TS-dynamic__", "__TS-fixed__ *" and "__TS-updated__ *".
```
python exp_thompson_lp.py
``` 
Additionally, for "__TS-episodic__ *" and "__TS-dynamic__ *", run the code 
```
python exp_thompson_lp_true.py
```
Experiment results will be stored in a directory called "data" within the same directory as a pickle file, such as "./data/XXXX.pickle" 
while the log file of the experiments are generated in a directory called "log" within the same directory as "./log/save_log.txt".  
To change experimental settings, edit parameters in the file "exp_thompson_lp.py" or "exp_thompson_lp_true.py". 
Changing the parameters allows you to simulate experiments for all of our settings including additional results in the appendix.   

## To plot regret 
Load the experimental results you have obtained and plot their regret in the notebook "Images_generator.ipynb".
