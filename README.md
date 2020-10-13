# DRLforPowerSystemRecovery
Master Thesis Project for DRL in power system restoration using renewables.  

The goal of this master thesis is to develop DRL techniques that can compute restoration sequences for power systems with high share of renewables. An example for a trained agent is shown below. 

![Agent in action](/rest-gym/plots/archive/sequence_39node_1m.gif)

This Git repo contains the code that accompanies the master thesis project. 


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

The initial state of the environment simulates the depleted state of a power system after a large scale blackout. The agent's task is to restore service for all loads. 

Given suitable weather conditions renewable energy generators can be used as black start resources, if cranking power is provided by storage units or the grid. The power output of renewable energy generators is determined by a randomly chosen point in time from real wheather data obtained from [renewables.ninja](https://www.renewables.ninja/) for a wind/solar power plant near Berlin, Germany. 


