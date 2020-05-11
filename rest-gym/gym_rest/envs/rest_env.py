# -*- coding: utf-8 -*-
"""
Simulates the simplified restoration environment. 
"""

# core modules
import logging.config
import random

# 3rd party modules
from gym import spaces
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
    Two instances of the same pandapower network are created: net1 and net2. 
    net1 is for reference, net2 is the network the agent interacts with. 
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("RestEnv - Version {}".format(self.__version__))
        
#        toggle between different max step limits to accelerate training
        
        self.TOTAL_TIME_STEPS = 49
#         case 14: 49
#         case 4: 29
#         case 39: 99


#       create networknet1 as reference network 
        
#        self.net1 = network_creator.create_4GS_PV_Wind_Storage()
        self.net1 = network_creator.create_case14_PV_Wind_Storage()
#        self.net1 = network_creator.create_case39_PV_Wind_Storage()

        
        pp_helpers.add_switches_to_lines(self.net1)
        self.net2 = copy.deepcopy(self.net1)
        pp.rundcpp(self.net1)
        pp.rundcpp(self.net2)
        
#        read scaling factors for sgens: 
        self.time = ""    
        self.scaling_wind = pd.read_csv("../time_series/ninja_wind_52.4475_13.2080_corrected.csv", skiprows=3, usecols=["time", "electricity"])
        self.scaling_pv = pd.read_csv("../time_series/ninja_pv_52.4475_13.2080_corrected.csv", skiprows=3, usecols=["electricity"])

#        define action duration in min. 
#        This is required to calculate the state of charge of batteries. 
#        TODO: implement realistic action duration times for each category
        
        self.action_duration = [1,1,1,1,1,1,1,1,1,1,1,1]

#        initiate network parameters
        
        self.total_loads = self.net1.res_load.p_mw.sum()
        self.n_storage = len(self.net2.storage.index)
        self.n_gen = len(self.net2.gen.index) - self.n_storage
        self.n_switch = len(self.net2.switch.index)
        self.n_varloads = len(self.net2.load.index)
        self.n_pv = len(self.net2.sgen[self.net2.sgen["type"]=="solar"])
        self.n_wind = len(self.net2.sgen[self.net2.sgen["type"]=="wind"])
        self.n_line = len(self.net2.line.index)


        self.pf_converges = False
        
#        network memory parameters
        
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
        
        self.non_slacks = self.net2.gen[self.net2.gen["slack"]==False].index
        self.connected_gens = sum(self.net2.res_gen.p_mw[self.non_slacks] > 0)
        self.connected_gens_memory = self.connected_gens
        
        self.connected_sgens = sum(self.net2.res_sgen.p_mw > 0)
        self.connected_sgens_memory = self.connected_sgens
        
        self.cranked_isolated_sgen = False
        self.is_net_restored = False
        
        self.curr_step = -1
        self.info = {}
        
#        get action and observation space for env
        
        self.action_space = self._getActionSpace("discrete") 

        self.observation_space = self._getObservationSpace()
        
#        m action categories 
        
        self.action_categories = [self.n_line, self.n_line, self.n_varloads, 
                                  self.n_varloads, self.n_gen, self.n_gen, 
                                  self.n_pv, self.n_pv, self.n_wind, self.n_wind, 
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
        (obs, reward, episode_over, info) as a tuple 
            obs (object) :
                the observation of the environment
            reward (float) :
                amount of reward achieved by the previous action. The goal is to
                maximize the received rewards
            episode_over (bool) :
                whether the episode is completed or not. If completed, the 
                environment is reset to its initial state and a new episode 
                begins.
            info (dict) :
                 diagnostic information useful for debugging. the info is 
                 invisible for the agent.
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
        """
        The effect of the agent's action on the environment is simulated. 
        
        As the action space is discrete, but the actions are tabulated in a (n,m) 
        matrix with m action categories with the corresponding action, the 
        received action is first transformed into the n,m format. 
        """
        self.action_episode_memory[self.curr_episode].append(action)
                  
#        transform discrete action into n,m action table format
#        m is the action category, n is the subordinated action of m
        
        m = 0
        n = action
        while n - self.action_categories[m] >= 0:
            n -= self.action_categories[m]
            m+=1
        
        if m == 0: 
#             activate line
            if pp_helpers.get_line_states(self.net2)[n] == 0:
                pp_helpers.switch_line(self.net2, n)
        elif m == 1: 
#             deactivate line
            if pp_helpers.get_line_states(self.net2)[n] == 1:
                pp_helpers.switch_line(self.net2, n)
        elif m == 2: 
#             activate load
            self.net2.load.scaling.at[n] = 1
        elif m == 3: 
#             deactivate load
            self.net2.load.scaling.at[n] = 0
        elif m == 4: 
#             activate gen and load at gen
            self.net2.gen.in_service.at[n] = 1
#             activate corresponding load? 
        elif m == 5: 
#             deactivate gen
            self.net2.gen.in_service.at[n] = 0
        elif m == 6: 
#             activate solar 
            solar_sgens = self.net2.sgen[self.net2.sgen["type"]=="solar"].index
            
#            check whether cranking power for sgen is available, and whether the 
#            sgen is cranked by an external grid or storage unit 
            
            (cranked_by_net, self.cranked_isolated_sgen) = pp_helpers.crank_sgen(self.net2, solar_sgens[n])
            if cranked_by_net or self.cranked_isolated_sgen:
                current_state = self.net2.sgen.in_service.at[solar_sgens[n]]

#                to avoid granting multiple rewards by repeatedly cranking the 
#                same sgen 
                if current_state: self.cranked_isolated_sgen = False
                
                self.net2.sgen.in_service.at[solar_sgens[n]] = 1
            
        elif m == 7: 
            # deactivate solar
            solar_sgens = self.net2.sgen[self.net2.sgen["type"]=="solar"].index
            self.net2.sgen.in_service.at[solar_sgens[n]] = 0
            
        elif m == 8:
            # activate wind
            wind_sgens = self.net2.sgen[self.net2.sgen["type"]=="wind"].index
            
#            check whether cranking power for sgen is available, and whether the 
#            sgen is cranked by an external grid or storage unit 
            
            (cranked_by_net, self.cranked_isolated_sgen) = pp_helpers.crank_sgen(self.net2, wind_sgens[n]) 
            if cranked_by_net or self.cranked_isolated_sgen:
                
#                to avoid granting multiple rewards by repeatedly cranking the 
#                same sgen 
                current_state = self.net2.sgen.in_service.at[wind_sgens[n]]
                if current_state: self.cranked_isolated_sgen = False
                
                self.net2.sgen.in_service.at[wind_sgens[n]] = 1
                
        elif m == 9: 
            # deactivate wind
            wind_sgens = self.net2.sgen[self.net2.sgen["type"]=="wind"].index
            self.net2.sgen.in_service.at[wind_sgens[n]] = 0
                
        elif m == 10: 
            # activate storage
            # assumption: storage unit requires 20 percent of its capacity to restart
            if self.net2.storage.soc_percent.at[n] >= .2:
                self.net2.storage.scaling.at[n] = 1
                b = self.net2.storage.bus.at[n]
                self.net2.gen[self.net2.gen["bus"]==b].in_service = 1
                
        elif m == 11: 
            # deactivate storage
            self.net2.storage.scaling.at[n] = 0
            bus = self.net2.storage.bus.at[n]
            self.net2.gen[self.net2.gen["bus"]==bus].in_service = 0
            
        # (inlcudes pf calculation)
        pp_helpers.scale_islanded_areas(self.net2)
        
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
#            only for debugging
            self.info = {}    

    def _get_reward(self):
        """
        computes the reward the agent received with the action according to the 
        reward function. 
        -----
        Returns for one step:
            +1000 if net is restored
            +50 for new loads supplied 
            -100 for lost loads
            +10 for restored lines
            -10 for disconnected lines
            +10 for connected gens
            -10 for disconnected gens
            +10 for connected sgens
            -10 for disconnected sgens
            +1 for activated storage units
            +100 if an isolated gen was provided with cranking power to restart
            
        if the accumulated reward is less or equal than 0, the agent is 
        additionally with -50 points.  
        """
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
        All transmission lines are set out of service, all elements in isolated 
        areas are set out of service. 
        
        The scaling factor of renewable energy resources is set in accordance to 
        real weather data at a random time of outage. 
        The solar and wind power plants used for reference are medium sized 
        plants located near Berlin, Germany. 
        
        The SOCs of storage units a randomly initialized randomly between .5 and 1

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        
#        copy reference network 
        self.net2 = copy.deepcopy(self.net1)

        # set all lines out of service
        self.net2.switch.closed = False
        
        self.cranked_isolated_sgen = False
        
        # scale sgens by determining random time of outage. 
        t = random.randint(0, len(self.scaling_wind)-1)
        self.net2.sgen.scaling[self.net2.sgen["type"]=="wind"] = self.scaling_wind.electricity.at[t]
        self.net2.sgen.scaling[self.net2.sgen["type"]=="solar"] = self.scaling_pv.electricity.at[t]
        self.time = self.scaling_wind.time.at[t]
        
        # set storages at random SOC between .5 and 1 
        self.net2.storage.soc_percent = random.randint(5,10)/10
        
        pp_helpers.set_unsupplied_areas_out_of_service(self.net2)
        
        self.curr_step = -1
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.initial_obs.append([])
        self.is_net_restored = False
        self.restoration_failed = False
        self.info = {}
        
        return self._get_state()

    def _render(self, save=False, draw=False, num='00', action_name = ''):
        """
        Forwards the call to render the environment to afunction of pp_helpers 
        module
        """
        pp_helpers.plot_render(self.net2, save, draw, num, action_name) 

    def _get_state(self):
        """Get the observation (vector containing a vectorized representation of
        what the agent observes).
        
        Returns
        -------
        concatenated ndarray containing:
            line states, 
            load supply level for each load, 
            power generation for each generator, 
            scaling factor of each PV plant, 
            power output of each PV plant, 
            scaling factor of each wind power plant, 
            power output of each wind power plant,
            power generation of each storage unit, and
            SOC of each storage unit. 
        
        """
        
        self.pf_converges = pp_helpers.run_dcpowerflow(self.net2, scale_gens=False, scale_loads=False)
        
        ob_line_switch = pp_helpers.get_line_states(self.net2)
        ob_load_supplied = self.net2.res_load.p_mw/self.net1.res_load.p_mw 
        ob_gen_power = self.net2.res_gen.p_mw[self.net2.gen["slack"]==False] > 0
        ob_pv_scaling = self.net2.sgen.scaling[self.net2.sgen["type"]=="solar"]
        ob_pv_power = self.net2.res_sgen.p_mw[self.net2.sgen["type"]=="solar"] > 0
        ob_wind_scaling = self.net2.sgen.scaling[self.net2.sgen["type"]=="wind"]
        ob_wind_power = self.net2.res_sgen.p_mw[self.net2.sgen["type"]=="wind"] > 0
        ob_storage_power = self.net2.res_storage.p_mw < 0
        ob_storage_soc = self.net2.storage.soc_percent
        
        ob = np.concatenate((ob_line_switch, ob_load_supplied, ob_gen_power, ob_pv_scaling, 
                             ob_pv_power, ob_wind_scaling, ob_wind_power, 
                             ob_storage_power, ob_storage_soc), None)
        
#       reset memory parameters
        
        if self.curr_step == -1: 
#             runaround; only update memory and parameters in this order if 
#             the function is called by reset() 
            self._update_parameters()
            self._update_memory()

#       definetely not the best way to do this... --> improve        
        return ob.reshape(len(ob),1)
    
    def _getActionSpace(self, shape="discrete"):
        """
        Get the action space of the environment. The only supported action space 
        shape is "discrete", as it is the most adaptable for different agents
        
        Returns 
        -------
        gym space representig the length of the discrete action space shape. 
        The length is determined by the number of controllable elements within 
        the environment. 
        Controllable elements: lines, laods, gens, PV plants, wind power plants,
        storage units. 
        Each element contributes with two actions: ON and OFF
        """
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
        """
        get the shape of the observation space
        
        Returns
        -------
        gym space element
        """
        n_ob = len(self._get_state())
        ob_space = spaces.Box(0,1, [n_ob,1])
        
        return ob_space
    
    def seed(self, seed):
        # TODO implement seed to make result reproducable
        return
        
    def check_restoration(self):
        """
        determines whether the restoration process has been successful (enough) 
        to grant a reward 
        
        Returns
        -------
        float between 0 and 1
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
        get the action list as a pandas dataframe, to assist when playing the 
        environment manually. The dataframe contains the names of the actions 
        and their durations in mins. 
        
        Returns
        -------
        DF with all actions and their durations
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
        """
        get matrix with all possible actions. m columns represent the action 
        categories, n lines the elements of each category
        
        Return
        ------
        matrix with all actions. 
        """
        single_action_matrix = [list(range(self.n_line)), list(range(self.n_varloads)), list(range(self.n_gen)),
                         list(range(self.n_pv)), list(range(self.n_wind)), list(range(self.n_storage))]
        
#        double each element to include both ON and OFF options
        double_action_matrix =  [ele for ele in single_action_matrix for i in range(2)]
        return double_action_matrix
    
    def get_observation_list(self): 
        """
        get list of all observation, to assist manually playing the environment.
        
        Return
        ------
        pandas Dataframe of all observations
        """
        ob_names = ["line switches", "loads supplied", "gen power" , "pv scaling", 
                    "pv powered", "wind scaling", "wind powered", "storage powered", 
                    "storage SOCs"]
        ob_n = [self.n_line, self.n_varloads, self.n_gen, self.n_pv, self.n_pv, 
                self.n_wind, self.n_wind, self.n_storage, self.n_storage]
        df = pd.DataFrame(list(zip(ob_names, ob_n)), columns=["name", "number"])
        return df
       
    
    
   
   
    
        

        
