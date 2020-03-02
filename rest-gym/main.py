# -*- coding: utf-8 -*-
"""
@author: Benedikt
"""

import gym
import time 
from stable_baselines.common.policies import MlpPolicy
#from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DQN

if 'Rest_Minimal-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['Rest_Minimal-v0']
    print("unregistered {} env".format('Rest_Minimal-v0'))

import gym_rest

env = gym.make('Rest_Minimal-v0')
# Vectorized environments allow to easily multiprocess training
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

env.reset()

#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='ppo2_rest_tensorboard/')
#model = PPO2(MlpPolicy, env, verbose=1)
model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='a2c_rest_tensorboard/')
#model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='dqn_rest_tensorboard')



# Train the agent

#use tensorboard with the following line in anaconda prompt: 
# Benedikt\Documents\Uni\MasterThesis\Code\project05_gym_continuous\rest-gym
# tensorboard --logdir C:\Users\Benedikt\Documents\Uni\MasterThesis\Code\project05_gym_continuous/rest-gym/a2c_rest_tensorboard/

t1 = time.time()
model.learn(total_timesteps=1000000)
t2 = time.time()
print("learning took {} seconds".format(t2-t1))

#model = A2C.load("a2c_4node_100000steps")

#pp_helpers.plot_sequence(env, model)

#action_list_func = env.env_method("_get_action_list")[0]

# save model
#model.save("A2C_39node_1000000steps")