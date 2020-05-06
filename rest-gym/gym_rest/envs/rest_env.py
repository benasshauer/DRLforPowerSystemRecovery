#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplified restoration environment.
"""

# core modules
import logging.config
#import math
#import pkg_resources
import random

# 3rd party modules
from gym import spaces
#import cfg_load
import gym
import numpy as np
import pandapower as pp 
import pandapower.topology as top
import pp_helpers 
import network_creator
import copy
import pandas as pd



class RestEnv(gym.Env):
    """
    Define a simple Restoration environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("RestEnv - Version {}".format(self.__version__))
        
        self.TOTAL_TIME_STEPS = 49
        # case 14: 49
        # case 4: 29
#         case 39: 99



#        self.net1 = network_creator.create_4GS_PV_Wind_Storage()
        self.net1 = network_creator.create_case14_PV_Wind_Storage()
#        self.net1 = network_creator.create_case39_PV_Wind_Storage()

#        self.net1 = pp_helpers.create_line_net()
#        self.net1["gen"].drop(0, inplace=True)
        
#        self.net1.gen.slack = True
#        pp.create_gen(self.net1, 1, p_mw=0, slack=True, name="bat")
#        pp.create_sgen(self.net1, 2, p_mw=100, name="PV")
        
        pp_helpers.add_switches_to_lines(self.net1)
        self.net2 = copy.deepcopy(self.net1)
        pp.rundcpp(self.net1)
        pp.rundcpp(self.net2)
        
        # read scaling factors for sgens: 
        self.time = ""    
        self.scaling_wind = pd.read_csv("../time_series/ninja_wind_52.4475_13.2080_corrected.csv", skiprows=3, usecols=["time", "electricity"])
        self.scaling_pv = pd.read_csv("../time_series/ninja_pv_52.4475_13.2080_corrected.csv", skiprows=3, usecols=["electricity"])

        # define action duration in min. 
        # This is required to calculate the state of charge of batteries. 
        self.action_duration = [1,1,1,1,1,1,1,1,1,1,1,1]

        self.total_loads = self.net1.res_load.p_mw.sum()
        
        self.n_storage = len(self.net2.storage.index)
        self.n_gen = len(self.net2.gen.index) - self.n_storage
        self.n_switch = len(self.net2.switch.index)
        self.n_varloads = len(self.net2.load.index)
#        self.n_storage = len(self.net2.storage.index)
        self.n_pv = len(self.net2.sgen[self.net2.sgen["type"]=="solar"])
        self.n_wind = len(self.net2.sgen[self.net2.sgen["type"]=="wind"])
        self.n_line = len(self.net2.line.index)
#        self.n_sgen = len(self.net2.sgen.index)


        self.pf_converges = False
        
        self.unsupplied_buses = sum(self.net2.res_load.p_mw/self.net1.res_load.p_mw < 1) 
        self.unsupplied_buses_memory = self.unsupplied_buses

        self.load_supply = self.net2.res_load.p_mw.sum()/self.total_loads
        self.load_supply_memory = self.load_supply
        
        self.n_load_supply = sum(self.net2.res_load.p_mw > 0)
        self.n_load_supply_memory = self.n_load_supply
        
        self.connected_lines = pp_helpers.get_line_states(self.net2).sum()
        self.connected_lines_memory = self.connected_lines
        
        self.storage_active = sum(self.net2.storage.scaling == 1)
        self.storage_active_memory = self.storage_active
#        if self.n_gen > 0:
#            self.connected_gens = self.net2.res_gen.p_mw[self.net2.gen["slack"]==False].sum()/self.net1.res_gen.p_mw.sum()
        self.non_slacks = self.net2.gen[self.net2.gen["slack"]==False].index
        self.connected_gens = sum(self.net2.res_gen.p_mw[self.non_slacks] > 0)
#            self.connected_gens = sum(self.net2.res_gen.p_mw[self.net2.gen["slack"]==False] > 0)
#        else: 
#            self.connected_gens = 0
            
        self.connected_gens_memory = self.connected_gens
        
#        self.connected_sgens = self.net2.res_sgen.p_mw.sum()/self.net1.res_sgen.p_mw.sum()
        self.connected_sgens = sum(self.net2.res_sgen.p_mw > 0)
        self.connected_sgens_memory = self.connected_sgens
        
        self.cranked_isolated_sgen = False
        self.is_net_restored = False
        
        self.curr_step = -1
        self.info = {}
        
        self.action_space = self._getActionSpace("discrete") #"Box" or "discrete"

        # power flow OK; (Knoten OK; Leitungen OK), alle Knoten verbunden,...  
#        self.observation_space = spaces.MultiDiscrete([2]*len(self.net2.bus))
        self.observation_space = self._getObservationSpace()
        
        self.action_categories = [self.n_line, self.n_line, self.n_varloads, self.n_varloads, self.n_gen, 
                                  self.n_gen, self.n_pv, self.n_pv, self.n_wind, self.n_wind, 
                                  self.n_storage, self.n_storage]
        

        
        # Store what the agent tried
        
        self.curr_episode = -1
        self.action_episode_memory = []
        self.initial_obs = []

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : array (int)

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.is_net_restored:
            raise RuntimeError("Episode is done")
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        self.initial_obs[self.curr_episode].append(tuple(ob))
        info = self.info
        done = self.is_net_restored or self.restoration_failed
        return ob, reward, done, info

    def _take_action(self, action):
        self.action_episode_memory[self.curr_episode].append(action)
                  
        # new method with action_matrix
        m = 0
        n = action
        while n - self.action_categories[m] >= 0:
            # das ist die falsche logik???
            n -= self.action_categories[m]
            m+=1
        
        if m == 0: 
            # activate line
            if pp_helpers.get_line_states(self.net2)[n] == 0:
                pp_helpers.switch_line(self.net2, n)
        elif m == 1: 
            # deactivate line
            if pp_helpers.get_line_states(self.net2)[n] == 1:
                pp_helpers.switch_line(self.net2, n)
        elif m == 2: 
            # activate load
            self.net2.load.scaling.at[n] = 1
        elif m == 3: 
            # deactivate load
            self.net2.load.scaling.at[n] = 0
        elif m == 4: 
            # activate gen and load at gen
            self.net2.gen.in_service.at[n] = 1
            # activate corresponding load? 
        elif m == 5: 
            # deactivate gen
            self.net2.gen.in_service.at[n] = 0
        elif m == 6: 
            # activate solar 
            solar_sgens = self.net2.sgen[self.net2.sgen["type"]=="solar"].index
            
            (cranked_by_net, self.cranked_isolated_sgen) = pp_helpers.crank_sgen(self.net2, solar_sgens[n])
            if cranked_by_net or self.cranked_isolated_sgen:
                current_state = self.net2.sgen.in_service.at[solar_sgens[n]]
                if current_state: self.cranked_isolated_sgen = False
                
                self.net2.sgen.in_service.at[solar_sgens[n]] = 1
            
        elif m == 7: 
            # deactivate solar
            solar_sgens = self.net2.sgen[self.net2.sgen["type"]=="solar"].index
            self.net2.sgen.in_service.at[solar_sgens[n]] = 0
            
        elif m == 8:
            # activate wind
            wind_sgens = self.net2.sgen[self.net2.sgen["type"]=="wind"].index
            
            (cranked_by_net, self.cranked_isolated_sgen) = pp_helpers.crank_sgen(self.net2, wind_sgens[n]) 
            if cranked_by_net or self.cranked_isolated_sgen:
                current_state = self.net2.sgen.in_service.at[wind_sgens[n]]
                if current_state: self.cranked_isolated_sgen = False
                
                self.net2.sgen.in_service.at[wind_sgens[n]] = 1
                
        elif m == 9: 
            # deactivate wind
            wind_sgens = self.net2.sgen[self.net2.sgen["type"]=="wind"].index
            
            self.net2.sgen.in_service.at[wind_sgens[n]] = 0
                
        elif m == 10: 
            # activate storage
            # storage unit requires 20 percent of its capacity to restart
            if self.net2.storage.soc_percent.at[n] >= .2:
                self.net2.storage.scaling.at[n] = 1
                b = self.net2.storage.bus.at[n]
                self.net2.gen[self.net2.gen["bus"]==b].in_service = 1
#                print("in_service = 1")
        elif m == 11: 
            # deactivate storage
            self.net2.storage.scaling.at[n] = 0
            bus = self.net2.storage.bus.at[n]
            self.net2.gen[self.net2.gen["bus"]==bus].in_service = 0
            
        # (inlcudes pf calculation)
        pp_helpers.scale_islanded_areas2(self.net2)
        
        pp_helpers.run_dcpowerflow(self.net2, scale_gens=False, scale_loads=False)
        
        pp_helpers.update_storage_SOC(self.net2, self.action_duration[m])
        
        
        self._update_memory()
        self._update_parameters()

#        if self.unsupplied_buses == 0:
#            self.is_net_restored = True
        if self.check_restoration() == 1:
            self.is_net_restored = True
        
        remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = (remaining_steps <= 0)
        
        self.restoration_failed = time_is_over and not self.is_net_restored
        if self.restoration_failed:
#            self.is_net_restored = True  # abuse this a bit
            self.info = {}    

    def _get_reward(self):
        
        reward = 0
        if self.is_net_restored:
            self.info = {}
            reward += 1000
        if self.n_load_supply > self.n_load_supply_memory:
            reward += 50
        if self.n_load_supply < self.n_load_supply_memory:
            reward -= 100
        if self.connected_lines > self.connected_lines_memory:
            reward += 10
        if self.connected_lines < self.connected_lines_memory:
            reward += -10
#        if self.connected_gens != self.connected_gens_memory:
#            reward += 10*(self.connected_gens - self.connected_gens_memory)
        if self.connected_gens > self.connected_gens_memory:
            reward += 10
        if self.connected_gens < self.connected_gens_memory:
            reward -= 10            
        if self.connected_sgens > self.connected_sgens_memory:
            reward += 10
        if self.connected_sgens < self.connected_sgens_memory:
            reward -= 10  
        if self.storage_active > self.storage_active_memory: 
            reward += 1
        if self.cranked_isolated_sgen: 
            reward += 100
            self.cranked_isolated_sgen = False
        if reward <= 0: reward += -50
        return reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.net2 = copy.deepcopy(self.net1)

        # old way with only a couple lines out of service
#        while len(top.unsupplied_buses(self.net2)) < 2:
#            disaster_loc = random.sample(list(self.net2.bus.index), 1)[0]
#            pp_helpers.destroy_grid3(self.net2, disaster_loc, 2) 

        # set all lines out of service
        self.net2.switch.closed = False
        
        self.cranked_isolated_sgen = False
        
        # scale sgens
        t = random.randint(0, len(self.scaling_wind)-1)
        self.net2.sgen.scaling[self.net2.sgen["type"]=="wind"] = self.scaling_wind.electricity.at[t]
        self.net2.sgen.scaling[self.net2.sgen["type"]=="solar"] = self.scaling_pv.electricity.at[t]
        self.time = self.scaling_wind.time.at[t]
        
#        pp_helpers.run_dcpowerflow(self.net2, scale_gens=False, scale_loads=False)
        # set storages at random SOC 
        self.net2.storage.soc_percent = random.randint(5,10)/10
        
        pp_helpers.set_unsupplied_areas_out_of_service(self.net2)
        
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.initial_obs.append([])
        self.is_net_restored = False
        self.restoration_failed = False
#        self.pf_converges = pp_helpers.run_dcpowerflow(self.net2, scale_gens=True, scale_loads=True)
        self.info = {}
        
#        self._update_parameters()
#        
#        self._update_memory()
        
        return self._get_state()

    def _render(self, save=False, draw=False, num='00', action_name = ''):
        pp_helpers.plot_render(self.net2, save, draw, num, action_name) 

    def _get_state(self):
        """Get the observation."""
#        ob = np.array([self.net2.gen.in_service.at[0], self.net2.switch.closed.at[0],
#                      self.net2.load.in_service.at[0]], int)
#        ob = np.array(self.net2.switch.closed, int)
        
        self.pf_converges = pp_helpers.run_dcpowerflow(self.net2, scale_gens=False, scale_loads=False)
        
        ob_line_switch = pp_helpers.get_line_states(self.net2)
#        ob_bus_voltage = np.nan_to_num(self.net2.res_bus.vm_pu)
        ob_load_supplied = self.net2.res_load.p_mw/self.net1.res_load.p_mw 
#        ob_gen_power =  self.net2.res_gen.p_mw/self.net1.res_gen.p_mw
        ob_gen_power = self.net2.res_gen.p_mw[self.net2.gen["slack"]==False] > 0
        ob_pv_scaling = self.net2.sgen.scaling[self.net2.sgen["type"]=="solar"]
        ob_pv_power = self.net2.res_sgen.p_mw[self.net2.sgen["type"]=="solar"] > 0
        ob_wind_scaling = self.net2.sgen.scaling[self.net2.sgen["type"]=="wind"]
        ob_wind_power = self.net2.res_sgen.p_mw[self.net2.sgen["type"]=="wind"] > 0
        ob_storage_power = self.net2.res_storage.p_mw < 0
        ob_storage_soc = self.net2.storage.soc_percent
#        ob = np.concatenate((ob_line_switch, ob_bus_voltage, ob_load_supplied), None)
        ob = np.concatenate((ob_line_switch, ob_load_supplied, ob_gen_power, ob_pv_scaling, 
                             ob_pv_power, ob_wind_scaling, ob_wind_power, 
                             ob_storage_power, ob_storage_soc), None)
        
        if self.curr_step == -1: 
            # runaround; only update memory and parameters in this order when 
            # the function is called by reset() 
            self._update_parameters()
            self._update_memory()
        
        return ob.reshape(len(ob),1)
    
    def _getActionSpace(self, shape="discrete"):
        """
        number of controls switches, positions of loads, variable loads and 
        gens
        either multidiscrete action space or single action space for different agents
        -> discrete action space is the most adaptable for different agents
        """
#        action_length = self.n_line + self.n_varloads + self.n_gen
        if shape == "multiDiscrete":
            env_space = spaces.MultiDiscrete([2]*self.n_switch)
        elif shape == "discrete":
#            env_space = spaces.Discrete(len(self.net2.line.in_service))
            action_discrete = 2*(self.n_line + self.n_varloads + self.n_gen + self.n_pv + self.n_wind + self.n_storage)
            env_space = spaces.Discrete(action_discrete)
        elif shape == "Box":
#            dim = self.n_line #+ self.n_varloads + self.n_gen
            env_space = spaces.Box(np.array([0,0,0,0,0,0]),np.array([self.n_line-1, 1,1,1,1, self.n_gen]))
#        elif shape == "Tuple":
#            env_space = spaces.Tuple([
#                    spaces.MultiDiscrete([2]*self.n_line),
#                    spaces.Box(0,1, [1,4])])
#        elif shape == "single": 
#            env_space = spaces.Tuple([
#                    spaces.Discrete(action_length),
#                    spaces.Box(0,1, [1,1])
#                    ])
        return env_space

    def _getObservationSpace(self):
#        ob_line_switch = spaces.MultiDiscrete([2]*len(self.net2.line.index))
#        ob_bus_voltage = spaces.Box(0,2,[1,len(self.net2.bus)])
#        ob_load_supplied = spaces.Box(0,1, [1,len(self.net2.load.index)])
#        ob_space = spaces.Tuple([
#                ob_line_switch, ob_bus_voltage, ob_load_supplied
#                ])
        n_ob = len(self._get_state())
        ob_space = spaces.Box(0,1, [n_ob,1])
        
        return ob_space
    
    def seed(self, seed):
        # TODO implement seed to make result reproducable
        # sets predefined area of the grid out of power, with predetermined 
        # Parameters for loads and storage...
#        random.seed(seed)
#        np.random.seed 
        return
        
    def check_restoration(self):
        """
        determines whether the restoration process has been successful (enough) 
        to grant a reward 
        [returns perc of loads restored?]
        """
        rest_perc = sum(self.net2.res_load.p_mw)/sum(self.net1.res_load.p_mw)
        return rest_perc
    
    def _update_memory(self):
        """
        saves current instance of restoration parameters
        """
        self.unsupplied_buses_memory = self.unsupplied_buses
        self.load_supply_memory = self.load_supply
        self.connected_lines_memory = self.connected_lines
        self.connected_gens_memory = self.connected_gens
        self.connected_sgens_memory = self.connected_sgens
        self.n_load_supply_memory = self.n_load_supply
        self.storage_active_memory = self.storage_active
        
    def _update_parameters(self):
        """
        updates restoration parameters
        """
        self.unsupplied_buses = sum(self.net2.res_load.p_mw/self.net1.res_load.p_mw < 1)
        self.load_supply = self.net2.res_load.p_mw.sum()/self.total_loads
        self.connected_lines = pp_helpers.get_line_states(self.net2).sum()
        self.connected_gens = sum(self.net2.res_gen.p_mw[self.non_slacks] > 0)
        self.connected_sgens = self.net2.res_sgen.p_mw.sum()/self.net1.res_sgen.p_mw.sum()
        self.n_load_supply = sum(self.net2.res_load.p_mw > 0)
        self.storage_active = sum(self.net2.storage.scaling == 1)
        
    def get_action_df(self):
        """
        returns the action list as a pandas dataframe
        """
        action_list = []
        action_names = ["activate line {}", "deactivate line {}", "activate load {}", "deactivate load {}", 
               "activate gen {}", "deactivate gen {}", "activate PV power plant {}", 
               "deactivate PV power plant {}", "activate wind power plant {}", "deactivate wind power plant {}", 
               "activate storage {}", "deactivate storage {}"]

        for m in range(len(self.action_categories)):
            for n in self._get_action_matrix()[m]: 
                action_list.append([action_names[m].format(n), self.action_duration[m]])    
                
        df = pd.DataFrame(action_list, columns=["name", "duration"]) 
        return df
    
    def _get_action_matrix(self):
        single_action_matrix = [list(range(self.n_line)), list(range(self.n_varloads)), list(range(self.n_gen)),
                         list(range(self.n_pv)), list(range(self.n_wind)), list(range(self.n_storage))]
        double_action_matrix =  [ele for ele in single_action_matrix for i in range(2)]
        return double_action_matrix
    
    def get_observation_list(self): 
        ob_names = ["line switches", "loads supplied", "gen power" , "pv scaling", 
                    "pv powered", "wind scaling", "wind powered", "storage powered", 
                    "storage SOCs"]
        ob_n = [self.n_line, self.n_varloads, self.n_gen, self.n_pv, self.n_pv, 
                self.n_wind, self.n_wind, self.n_storage, self.n_storage]
        df = pd.DataFrame(list(zip(ob_names, ob_n)), columns=["name", "number"])
        return df
       
    
    
   
   
    
        

        
