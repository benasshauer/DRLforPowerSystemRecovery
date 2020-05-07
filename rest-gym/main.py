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
import pp_helpers

if 'Restoration_Env-v0' in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs['Restoration_Env-v0']
    print("unregistered {} env".format('Restoration_Env-v0'))

import gym_rest

env = gym.make('Restoration_Env-v0')
env_dummy = gym.make('Restoration_Env-v0')

# Vectorized environments allow to easily multiprocess training
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

env.reset()

### set model

#model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='ppo2_rest_tensorboard/')
#model = PPO2(MlpPolicy, env, verbose=1)
#model = A2C(MlpPolicy, env, verbose=1, tensorboard_log='a2c_rest_tensorboard/')
#model = DQN(MlpPolicy, env, verbose=1, tensorboard_log='dqn_rest_tensorboard')
#note: DQN training requires a different set of MlpPolicy 

#### or load a pretrained model

model = A2C.load("trained_agents/A2C_14node_200000steps_newrewards", env=env)

### Train the agent

#t1 = time.time()
#model.learn(total_timesteps=250000)
#t2 = time.time()
#print("learning took {} seconds".format(t2-t1))

### enjoy the trained agent

obs = env.reset()
action = model.predict(obs)
obs, reward, done, info = env.step(action)

### save model

#model.save("A2C_14node_200000steps")

### plot a restoration sequence 

#pp_helpers.plot_sequence(env, model)

# get an action list of the env to understand the different actions

action_list = env.env_method("get_action_df")
#pp_helpers.plot_restoration_process(env, model, action_list)


