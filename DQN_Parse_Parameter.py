import argparse
import pandas as pd
import torch as T

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
########### data demand ##########
training_raw5= pd.read_csv('data/train/capacity.csv')
df5=pd.DataFrame(training_raw5 , columns=["DEMAND"])
dff5=pd.DataFrame(data=df5)
dff5.to_csv('data_out/n_demand.csv', mode='a' , index=False , header=False)

######### 1000 OR 5000 FOR TRAINING 
######### 2022 FOR TESTING PRICE_TEST

def parse_args():
    parser = argparse.ArgumentParser("MADDPG experiments for EV charging scheduling")

    # core training parameters
    parser.add_argument("--device", default=device, help="training device ")
    parser.add_argument("--episodes", type=int, default=2050, help="num of agents")
    parser.add_argument("--max_episode_steps", type=int, default=24, help="max steps for each episode")
    parser.add_argument("--target_replace_iter",type=int, default=5, help="target net update interval")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes from buffer")
    parser.add_argument("--memory_size", type=int, default=100000, help="number of data stored in the memory")
    # parser.add_argument("--log", type=str, default="./log/epoch_{}/{}", help="tensorboard log dir")

    parser.add_argument("--demand_file", type=str, default="data_out/n_demand.csv", help="ele demand file dir")
    parser.add_argument("--price_train", type=str, default="data_out/price_train.csv", help="ele price file train dir")
    parser.add_argument("--price_test", type=str, default="data_out/price_test.csv", help="ele price file test dir")
    parser.add_argument("--energy_train", type=str, default="data_out/energy_train.csv", help="ele energy file train dir")
    parser.add_argument("--energy_test", type=str, default="data_out/energy_test.csv", help="ele energy file test dir")


    parser.add_argument("--price_source", type=str, default="lstm", help="ele price used for training, select in: random, real, lstm")


    # networks settings
    parser.add_argument("--fc_1", type=int, default=64, help="dims of the first full connected layers")
    parser.add_argument("--fc_2", type=int, default=64, help="dims of the second full connected layers")
    parser.add_argument("--fc_3", type=int, default=64, help="dims of the second full connected layers")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for adam optimizer in DQN")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for agent to foresee the state")
    parser.add_argument("--tau", type=int, default=0.01, help="randomness of buffer sampling in soft update")
    parser.add_argument("--epsilon", type=int, default=1.0 ,help="greedy policy factor")
    parser.add_argument("--al", type=int, default=0.1 ,help="current values of Q-Table at each step")

    parser.add_argument("--epsilon_decay", type=int, default=0.995 ,help="decay greedy policy factor")
    parser.add_argument("--epsilon_min", type=int, default=0.01 ,help="minimum greedy policy factor")

    # env settings
    parser.add_argument("--max_battery_capacity", type=int, default=24 , help="max of the EV battery capacity")
    parser.add_argument("--min_battery_capacity", type=int, default=1 , help="min of the EV battery capacity")
    parser.add_argument("--target_battery_capacity", type=int, default=24 , help="max of the EV battery capacity")
    parser.add_argument("--fixed_charge_cost", type=int, default=2.0 , help="max of the EV battery capacity")

    parser.add_argument("--max_charge_power", type=float, default=0.1)
    parser.add_argument("--min_charge_power", type=float, default=-0.9)
    parser.add_argument("--base_load", type=float, default=1.0, help="basic power system load")
    parser.add_argument("--load_restriction", type=float, default=0.06, help="load reward limited by the persentage of charging load in base load")
    parser.add_argument("--price_factor", type=float, default=1, help="discount of price reward")
    parser.add_argument("--energy_factor", type=float, default=1, help="discount of energy reward")
    parser.add_argument("--demand_factor", type=float, default=1, help="discount of demand reward")
    parser.add_argument("--export_factor", type=float, default=1, help="discount of export reward")
    parser.add_argument("--import_factor", type=float, default=1, help="discount of import reward")

    parser.add_argument("--load_factor", type=float, default=3, help="discount of load reward")
    parser.add_argument("--soc_factor", type=float, default=1, help="discount of soc reward")

    return parser.parse_args()