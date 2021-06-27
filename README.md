# cs295-RL-MiniProject
This is the repo for the Mini-Project for the class of CS 295-Reinfocement Learning.

## Files
The files are as folows:
- DDPG*.py TD3.py contain the models for DDPG, TD3, and their variants.
- train-DDPG*.py contain the codes to train the models
- plot_ddpg*.py contain the coddes to plot the models' results.
- plots/* - contain the plots of the resulting runs
- results/* - contain the (output) data of the resulting runs

## Executing code
Generally to run an experiment, just do `python train*. For example,

    python train-TD3.py 

trains using the TD3 model.  Then to plot the output do ` python plot_*.py`. For example,

    python plot_td3.py

