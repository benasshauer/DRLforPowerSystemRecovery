# DRLforPowerSystemRecovery
Master Thesis Project for DRL in power system restoration using renewables 

The goal of this master thesis is to develop DRL techniques that can compute restoration sequences for power systems with high share of renewables. An example for a trained agent is shown below. 
![Agent in action](/rest-gym/plots/archive/sequence_14node_250k.gif)
This Git repo contains the code that accomanies the master thesis project. 
It can be installed as a PIP package which is an OpenAI environment for
simulating a power system restoration process. 

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

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_rest

env = gym.make('RestEnv-v0')
```



## The Environment
