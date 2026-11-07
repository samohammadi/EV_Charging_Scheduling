import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
import pandas as pd
#import LSTM_GPT_ENERGY
from DQN_Parse_Parameter import parse_args

args = parse_args()

# EV agent = [soc, soc_ideal, start_time, end_time]
# base_load = 1.0
# max_battery_capacity = 1.5
# min_charge_power = -10
# max_charge_power = 10
# load_restriction = 0.3
# load_factor = 0.7
# soc_factor = 0.5


# ver 1.4 for DQN
class Energy(gym.Env):
    

    def __init__(self, n_agents, max_time_steps=24):
        self.n_agents = n_agents
        self.date = 0
        self._time_step = 0  # time pointer
        self._max_time_steps = max_time_steps

        # load ele price file
        
        self.file = np.loadtxt(args.energy_train,
                               delimiter=",",
                               encoding="UTF-8")
        #self.lstm_model = torch.load("lstm/lstm_2000.pt")
        #self.predict_price = lstm.predict(self.lstm_model).flatten()
       
        self.ele_energy = self._update_ele_Energy_LSTM()

# action means charging power, range [P_min, P_max], but discrete, 3 for cumulative charging cost and 7 for cumulative reward
        self.action_space = spaces.Discrete(7) 
        # self.action_space = spaces.Box(
        #     low=args.min_charge_power, high=args.max_charge_power,
        #     shape=(1, ))  # action means charging power, range [P_min, P_max]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_agents,1))  # soc
        
    def step(self, action):
        obs_n = []
        reward = 0
        done_n = []
        info_n = 0

        # reward_n = [0 for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            # charge EV
            #if (self._time_step >= self.start_time_list[i]
                   # and self._time_step <= self.end_time_list[i]):
            self._update_agent_state(i, action)
            reward = self._get_reward(i, action)
            info_n= self.cumulative(action)
            # if EV leaves, task done
            if self._time_step > self.end_time_list[i]:
                self.done[i] = True
            else:
                self.done[i] = False

        # obs = [soc_1, soc_2,..., soc_n_agents, ele_price]
        # obs[n_agents] = ele_price
        obs_n = self.soc_list
        done_n = self.done

######## update ele price  for next time step
        self.ele_energy = self._update_ele_Energy_LSTM()
        #print("Time:{}".format(self._time_step),"energy :{:.2f}".format(self.ele_energy))
        d={'Time':self._time_step ,'Energy':self.ele_energy}
        df=pd.DataFrame(data=d, index=[1])
        df.to_csv('data_result/Energy_file.csv', mode='a', index=False, header=False)
      
        self._time_step += 1
        #print("Time:{}".format(self._time_step),"price :{}".format(self.ele_energy))

        return obs_n, reward, done_n, info_n
    def cumulative(self,action):
        #print('action:{}'.format(action))
        p=round(self.ele_energy,1)
        pri = p * action
        #print('action*price:{}'.format(round(pri,1)))
        return round(pri,1)
    



    def reset(self):
        self._time_step = 0
        self.done = [False for i in range(self.n_agents)]

        self.soc_list = []
        self.soc_ideal_list = []
        self.start_time_list = []
        self.end_time_list = []

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

            # record all agent states
            self.soc_list.append(soc)
            self.soc_ideal_list.append(soc_ideal)
            self.start_time_list.append(start_time)
            self.end_time_list.append(end_time)

        # obs = [soc_1, soc_2,..., soc_n_agents, ele_price]
        observation = self.soc_list.copy()
        obs = np.array(observation)
        # obs = np.append(obs, self.ele_price)
        # obs = np.concatenate((observation, price), axis=1)

        return obs

    def _update_agent_state(self,agent,  action):
        # this func is used to update EV soc
        self.soc_list[agent] += action  # charge EV with current Power=action

    def _get_reward(self,agent,  action):
        energy_reward = args.energy_factor * self.ele_energy * action

        if self._time_step == self.end_time_list[agent]:
            soc_reward = args.soc_factor * (args.max_battery_capacity ** action) - 0.01 * (self.soc_ideal_list[agent] - self.soc_list[agent]) ** 2
        else:
            soc_reward = args.soc_factor * (args.max_battery_capacity ** action)

        return energy_reward - soc_reward
        



    def _update_ele_Energy_LSTM(self):
        ene = self.date * 288 + self._time_step * 12
        val_e =np.mean(self.file[ene:ene + 12])

        if self._time_step % 24 == 0:
            self.date += 1
            if self.date % 30 == 0:
                self.date = 0
        #print(('ene:{}'.format(val_e)))
        return val_e