import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
import pandas as pd
from DQN_Parse_Parameter import parse_args
import pandas as pd
args = parse_args()

class Demand(gym.Env):
    
    
    def __init__(self, n_agents, max_time_steps=23):
        self.n_agents = n_agents
        self.date = 0
        self._time_step = 0  # time pointer
        self._max_time_steps = max_time_steps

        # load ele price file
        self.file = np.loadtxt(args.demand_file,
                               delimiter=",",
                               encoding="UTF-8")
        self.ele_demand = self._update_ele_demand_real()

        # action means charging power, range [P_min, P_max], but discrete, 3 for cumulative charging cost and 7 for cumulative reward

        self.action_space = spaces.Discrete(7) 
        self.observation_space = spaces.Box(
            low=0.0, high=2.0,
            shape=(self.n_agents,1))  # soc
    
 
    def step4(self, action):
        obs_n = []
        reward = 0
        done_n = []
        info_n = {"n": []}
       
        for i in range(self.n_agents):
            # charge EV
            #if (self._time_step >= self.start_time_list[i]
                    #and self._time_step <= self.end_time_list[i]):
            self._update_agent_state(i, action)
            reward = self._get_reward4(i, action)

            # if EV leaves, task done
            if self._time_step > self.end_time_list[i]:
                self.done[i] = True
            else:
                self.done[i] = False
        obs_n = self.state_charging
        done_n = self.done
        # update ele demand  for next time step
        self.ele_demand = self._update_ele_demand_real()
        self._time_step += 1
        #print("      ","Demand :{}".format(self.ele_demand))
        d={ 'Demand':self.ele_demand}
        df=pd.DataFrame(data=d, index=[1])
        df.to_csv('data_result/Demand_file.csv', mode='a', index=False, header=False)
        return obs_n, reward, done_n, info_n
    
    def reset4(self):
        self._time_step = 1
        self.done = [False for i in range(self.n_agents)]

        self.soc_list = []
        self.soc_ideal_list = []
        self.start_time_list = []
        self.end_time_list = []
        self.state_charging=[]

        # init each agent
        for i in range(self.n_agents):
            start_time = np.random.randint(23, high=None, size=None,
                                           dtype="l")  # init start_time
            end_time = np.random.randint(start_time,
                                         high=23,
                                         size=None,
                                         dtype="l")  # init end_time
            soc_ideal = np.random.normal(loc=0.5, scale=0.16,
                                         size=None)  # init ideal soc ideal
            soc = np.random.normal(loc=0.5, scale=0.16, size=None)  # init soc
            state_charging = np.random.randint(4, high=None, size=None,
                                           dtype="l")  # state
            # record all agent states
            self.soc_list.append(soc)
            self.soc_ideal_list.append(soc_ideal)
            self.start_time_list.append(start_time)
            self.end_time_list.append(end_time)
            self.state_charging.append(state_charging)

        observation = self.soc_list.copy()
        obs = np.array(observation)
        return obs

    def _update_agent_state(self, agent, action):
        # this func is used to update EV soc
        self.soc_list[agent] += action  # charge EV with current Power=action

    def _get_reward4(self,agent, action):
        # demand reward
        demand_reward = args.demand_factor * self.ele_demand * action
         
        if self._time_step == self.end_time_list[agent]:
            soc_reward = args.soc_factor * (args.max_battery_capacity ** action)
            - 0.01*( pow((self.soc_ideal_list[agent] - self.soc_list[agent]), 2))
        else:
            soc_reward = args.soc_factor * ( args.max_battery_capacity ** action)

        reward=demand_reward  - soc_reward
        return reward

  
    def _update_ele_demand_real(self):
        # load real ele demand 
        if self._time_step % 24 == 0:
            self.date += 1
            if self.date % 30 == 0:
                self.date = 0
        z=self.date * self._max_time_steps + self._time_step
        #print(('dem:{}'.format(self.file[self.date * self._max_time_steps + self._time_step] )))
        return self.file[self.date * self._max_time_steps + self._time_step] 
        





   