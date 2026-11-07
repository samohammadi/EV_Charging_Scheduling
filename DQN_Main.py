# Cleaned-up version of your DQN code

import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from Prediction_Price import Price
from Prediction_Energy import Energy
from Demand_Forecasting import Demand
from Prediction_Price_Test import Price_test
from DQN_Parse_Parameter import parse_args

# Parse arguments
args = parse_args()

# Hyperparameters
BATCH_SIZE = args.batch_size
LR = args.lr
EPSILON = args.epsilon
GAMMA = args.gamma
TARGET_REPLACE_ITER = args.target_replace_iter
MEMORY_CAPACITY = args.memory_size
N_EPISODES = args.episodes
MAX_TIME_STEPS = args.max_episode_steps
ALPHA = args.al
EPSILON_MIN = args.epsilon_min
EPSILON_DECAY = args.epsilon_decay
# Environments
pri = Price(1)
ene = Energy(1)
dem = Demand(1)
pri_t = Price_test(1)

#actio for training , charging and discharging
#power_list = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]

#action for testing and charging
power_list =[0.01, 0.02, 0.03]


# Action and State space
N_ACTIONS = pri_t.action_space.n
N_STATES = pri_t.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(pri_t.action_space.sample(), int) else pri_t.action_space.sample().shape

FC_1 = args.fc_1
FC_2 = args.fc_2
FC_3 = args.fc_3

# TensorBoard writer
writer = SummaryWriter(f"log/DQN/{N_EPISODES}/{time.strftime('%m%d_%H%M')}")

# Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, FC_1)
        self.fc2 = nn.Linear(FC_1, FC_2)
        self.fc3 = nn.Linear(FC_2, FC_3)
        self.out = nn.Linear(FC_3, N_ACTIONS)

        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.zero_()

        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.zero_()

        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.zero_()

        self.out.weight.data.normal_(0, 0.1)
        self.out.bias.data.zero_()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

Q_table = {}

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss = 0
    def choose_action(self, x):

        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            action = np.random.randint(0, N_ACTIONS)
            #action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        else:   # random
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index

        action = power_list[action]
        #print(action)
        return action


    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        loss = self.loss_func(q_eval, q_target)
        self.loss = loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        df = pd.DataFrame({'Loss_function': [round(float(loss), 5)]})
        df.to_csv('data_result/Loss_function.csv', mode='a', index=False, header=False)

    def choose_action_q(state, EPSILON):
        state = tuple(state) if isinstance(state, (list, np.ndarray)) else (state,)
        if state not in Q_table:
            Q_table[state] = {a: 0.0 for a in power_list}
        return random.choice(power_list) if random.uniform(0, 1) < EPSILON else max(Q_table[state], key=Q_table[state].get)

    def update_q_table(self,state, action, reward, next_state, gamma, alpha):
        state = tuple(state) if isinstance(state, (list, np.ndarray)) else (state,)
        next_state = tuple(next_state) if isinstance(next_state, (list, np.ndarray)) else (next_state,)

        if next_state not in Q_table:
            Q_table[next_state] = {a: 0.0 for a in power_list}
        if state not in Q_table:
            Q_table[state] = {a: 0.0 for a in power_list}

        best_next_action = max(Q_table[next_state], key=Q_table[next_state].get)
        Q_table[state][action] += alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])

######################################################### Training Loop #######################################################################

dqn = DQN()
print("\nModel Init...")

episode_rewards = []
average_q_values = []
#epsilon = 1.0
prev_action = 0


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
episode_costs = []
for i_episode in range(N_EPISODES):
    s = pri.reset()
    s1 = ene.reset()
    s2 = dem.reset4()
    s3 = pri_t.reset_test()

    print(f'-------------Episode: {i_episode}-------------')
    # caculate cumulative reward 
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    # calculate cumulative price
    #epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    total_reward = 0
    total_steps = 0
    cum = 0

    for t in range(MAX_TIME_STEPS):
        a = dqn.choose_action(s)
        b = dqn.choose_action(s1)
        c = dqn.choose_action(s2)
        act=a

        
        pd.DataFrame([[act]]).to_csv('data_result/actions.csv', mode='a', index=False, header=False)

        s_, r1, done, info = pri.step(a)
        s1_, r2, done, info = ene.step(b)
        s2_, r3, done, info = dem.step4(c)
        s3_, r4, done, cum_p = pri_t.step_test(a)



        dqn.store_transition(s  ,a, r1, np.int32(s_))
        dqn.store_transition(s1 ,b, r2, np.int32(s1_))
        dqn.store_transition(s2 ,c, r3, np.int32(s2_))
       # dqn.store_transition(s3 ,c, r3, np.int32(s3_))

############################################## Q table #########################################################################################
       # action = a
       # next_state, reward, _, _ = pri.step(action)
        
        n_s = np.random.choice(4)
        dqn.update_q_table(s_, a, r1, n_s, GAMMA, ALPHA)
       # total_reward += reward

        #print(f"---------------------------\nEpisode {i_episode+1}, Step {t+1}\nState: {s_}, Action: {a}, Reward: {r1}, Next State: {n_s}")
################################################################################################################################################

################################################## Save Q-table ################################################################################
        pd.DataFrame({'State': [s_], 'Action': [a], 'Reward': [r1], 'Next State': [n_s]}).to_csv('data_result/Q_Table.csv', mode='a', index=False, header=False)

        all_qs = [q for state in Q_table.values() for q in state.values()]
        average_q = np.mean(all_qs) if all_qs else 0
        average_q_values.append(average_q)
################################################################################################################################################ 

################################################# reward ############################################################################################################
        r1 += -0.5 * abs(act - prev_action)
        prev_action = act

        for state, act_val, rew in zip([s, s1, s2], [a, b, c], [r1, r2, r3]):
            dqn.store_transition(state, act_val, rew, np.int32(s_))

        if total_steps % 100 == 0:
            dqn.learn()

        pd.DataFrame({'t': [r1], 'a': [act]}).to_csv('data_result/reward_on_hour.csv', mode='a', index=False, header=False)

        total_reward += r1
        pd.DataFrame({'t': [total_reward]}).to_csv('data_result/reward_on_24r.csv', mode='a', index=False, header=False)
        pd.DataFrame({'Total_reward': [total_reward]}).to_csv('data_result/total_reward.csv', mode='a', index=False, header=False)
################################################################################################################################################ 

##################################################### PRICE ##############################################################################################################################
        cum += round(cum_p, 5)
        total_steps += 1

    pd.DataFrame({'Cumulative_Price': [round(cum, 5)]}).to_csv('data_result/Cumulative_Price.csv', mode='a', index=False, header=False)
    episode_rewards.append(total_reward)
pri.print_soc_results()
################################################################################################################################################
# ==========================================
# FINAL TEST COST RESULTS
# ==========================================

if len(episode_costs) > 0:

    average_daily_cost = np.mean(episode_costs)

    print("\n========== TEST COST RESULTS ==========")
    print("Number of test episodes:", len(episode_costs))
    print("Average Daily Cost for 100 EVs:", round(average_daily_cost, 5))
    print("Minimum Daily Cost:", round(np.min(episode_costs), 5))
    print("Maximum Daily Cost:", round(np.max(episode_costs), 5))

    results = pd.DataFrame({
        'Episode': range(1, len(episode_costs) + 1),
        'Daily_Cost_100_EV': episode_costs
    })

    results.to_csv(
        'data_result/Daily_Cost_100_EV.csv',
        index=False
    )

else:
    print("ERROR: No episode costs were recorded!")

############################# Plot Episode Rewards ######################################################################################################
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(moving_average(episode_rewards, 50), label='Moving Average (50 episodes)', color='orange')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards with Moving Average')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('data_result/episode_rewards_plot.png')
plt.show()
################################################################################################################################################


##################################################
# updating weights and biass
import pandas as pd

weights_biases = {}
for name, param in dqn.eval_net.named_parameters():
    if param.requires_grad:
        weights_biases[name] = param.data.cpu().numpy().flatten()

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in weights_biases.items() ]))
df.to_csv("data_result/weights_biases.csv", index=False)
print("Weights and biases saved to data_result/weights_biases.csv")
##############################################################################
import numpy as np

soc_values = np.array(pri.departure_soc_history)

if len(soc_values) > 0:

    print("\n========== SOC TEST RESULTS ==========")

    print("Number of departed EVs:", len(soc_values))

    print("Mean SOC (%):",
          np.mean(soc_values) * 100)

    print("Minimum SOC (%):",
          np.min(soc_values) * 100)

    print("Maximum SOC (%):",
          np.max(soc_values) * 100)

    print("Standard Deviation (%):",
          np.std(soc_values) * 100)

    print("SOC below 20%:",
          np.sum(soc_values < 0.2))

    print("SOC between 20% and 90%:",
          np.sum((soc_values >= 0.2) & (soc_values <= 0.9)))

    print("SOC between 90% and 100%:",
          np.sum((soc_values > 0.9) & (soc_values <= 1.0)))

    print("SOC above 100%:",
          np.sum(soc_values > 1.0))

    print("======================================")

else:
    print("No departure SOC records available.")