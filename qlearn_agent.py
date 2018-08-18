# by ahnchive@gmail.com, 07/2018
# code modified from the ‘q_learning_agent.py’ by RLcode team (https://github.com/rlcode)


import numpy as np
import random
from collections import defaultdict
from gridworld import WindyGW
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

''' Q-learning control algorithm (Sutton & Barto, 2018)

Loop for each episode
    Initialize S
    Loop for each step of episode
        Choose A from S using policy derived from Q (epsilon-greedy) (off-policy)
        Take S, A, and observe R', S'
        Q(s,a) <- Q(s,a) + alpha * (R' + gamma * max_a Q (s',a) - Q (s,a))
        S <- S'
    Until S is terminal

'''


class QlearnAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # Qlearning (SARS) : Q <- Q + alpha * (R' + gamma* max_a Q(s',a) - Q)
    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q  = max(self.q_table[next_state])
        new_q = (current_q + self.learning_rate *
                 (reward + self.discount_factor * next_max_q - current_q))
        self.q_table[state][action] = new_q

    # choose A from S using policy derived from Q (epsilon-greedy)
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # random action choice
            action = np.random.choice(self.actions)
        else:
            # q max action choice
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


if __name__ == "__main__":
    env = WindyGW()
    agent = QlearnAgent(actions=env.possible_actions)
    step = []

    # loop for each episode
    for episode in range(200):
        #print("episode %d" % episode)

        # initialize S
        state = env.start

        time = 0

        # loop for each step of episode
        while True:

            time += 1

            # choose A from S using policy derived from Q (epsilon-greedy) (off-policy)
            action = agent.get_action(str(state))

            # take action A, observe R', S'
            reward = env.get_reward(state, action)
            next_state = env.state_after_action(state, action)

            # Qlearning (SARS) : Q <- Q + alpha * (R' + gamma* max_a Q(s',a) - Q)
            agent.learn(str(state), action, reward, str(next_state))

            state = next_state

            # (opt) print q_table
            # print (agent.q_table)

            # until S is terminal
            if state == env.goal:
                break

        print(time)
        step.append(time)


    # draw the number of steps per episode
    plt.figure()
    plt.plot(step)
    plt.xlabel('episode')
    plt.ylabel('num of steps')
    plt.show()

    # print q table
    # print(agent.q_table)

    # print the optimal policy
    print('Optimal policy is:', end='')

    optimal = []
    state = env.start
    while True:
        state_action = agent.q_table[str(state)]
        action = agent.arg_max(state_action)
        next_state = env.state_after_action(state, action)
        state = next_state

        if action == 0 :
            print("L", end='')
        elif action == 1 :
            print("R", end=''),
        elif action == 2 :
            print("U", end=''),
        elif action == 3:
            print("D", end=''),

        if state == env.goal:
            print("G")
            break
