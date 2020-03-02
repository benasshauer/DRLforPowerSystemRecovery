# DRLforPowerSystemRecovery
Master Thesis Project for DRL in power system restoration using renewables 

The goal of this master thesis is to develop DRL techniques that can compute restoration sequences for power systems with high share of renewables. An example for a trained agent is shown below. 
![Agent in action](/rest-gym/plots/archive/sequence_14node_250k.gif)

This Git repo contains the code that accomanies the master thesis project. 


## Dependencies ##
* **gym** for the RL agent setup
* **pandapower** for power system simulation 
* **stable_baselines** for DRL algorithms
* matplotlib
* pandas
* numpy
* networkx
* PIL
* seaborn






## Installation

This repo can be installed as a pip package. 
Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```


## The Environment
