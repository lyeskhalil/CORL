Deep RL for Online Combinatorial Optimization
==============================

Deep Policies for Online Bipartite Matching: A Reinforcement Learning Approach

NOTE: The repository is currently being re-organized for easier use.

Setting up
------------
Clone the repo and create a new python environment for you to run this project.
1. Install Pytorch Geometric (see [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)).
2. Instal and obtain a license for Gurobi (see [here](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html)).
3. Install all the other libraries listed in requirements.txt. This can be done by running:
`pip install -r requirements.txt`.


Running the Code
------------
run the file pipeline.py to do the following:
- generate a dataset (see generate_data())
- train a list of model on the dataset (see train_model())
- evaluate the models (see evaluate_model())

If you wish to only do one of the above, comment out the right function at the button of the file. 

You will need to open the file pipeline.py and change the flags on top of the file to generete the desired datasets, set model specifications, etc. See options.py for a detailed description of the flags. 

The dir "Dataset" includes a toy dataset of 50 trianing + 10 validation + 10 evaluation data points, where each datapoint is 10by30 bipartite graph(see the flags in pipeline for other specifications). Each data file "data_x.pt" also includes the optimal solution as well as the value of the optimal solution.

Code
--------
**Data Generation**: The dir "data" contains the base graph for gMission and MovieLens datasets in raw .txt format. data/generate_data.py produces datasets of bipartite graphs from these base graphs as well as from synthetic BA and ER graph generation schemes. 

**Environments**: The environment is implemented under the dir `problem_state` for  4 problems, namely obme, e-obm, adwords, and osbm.

**Models**: The models and the greedy baseline can be found under dir `policy`

