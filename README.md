corl
==============================

Deep Policies for Online Bipartite Matching: A Reinforcement Learning Approach

**Environment Details**:

Setting up
------------
clone the repo and create a new python environment for you to run this project.

To get started, install all the libraries listed in requirements.txt. This can be done by running.
`pip install -r requirements.txt`.

You will also need to install gurobi to run the poject (see [here](https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html)).

Running the Code
------------
run the file pipeline.py to do the following:
- generate a dataset (see generate_data())
- train a list of model on the dataset (see train_model())
- evalate the models (see evaluate_model())
- 
If you wish to only do one of the above, comment out the right function at the button of the file. 
You will need to open the file pipeline.py and change the flags on top of the file to generete the desired datasets, set model specifications, etc. See options.py for a detailed description of the flags. The current flags will prodcue ...

Code
--------


Shout-Outs
--------

<p><small>Original project template based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
