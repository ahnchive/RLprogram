# by ahnchive@gmail.com, 07/2018
# code modified from the ‘q_learning_agent.py’ by RLcode team (https://github.com/rlcode)

import numpy as np
import random
from collections import defaultdict
from gridworld import WindyGW
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

''' Double Q-learning control algorithm (Sutton & Barto, 2018)

Loop for each episode
    Initialize S
    Loop for each step of episode
        Choose A from S using policy epsilon-greedy derived from Q1 + Q2
        Take S, A, and observe R', S'
        With 0.5 probability:
            Q1(s,a) <- Q1(s,a) + alpha * (R' + gamma * Q2 (s',arg_max_a Q1(s',a)) - Q1 (s,a))
        else:
            Q2(s,a) <- Q2(s,a) + alpha * (R' + gamma * Q1 (s',arg_max_a Q2(s',a)) - Q2 (s,a))
        S<- S'
    Until S is terminal

'''


class DoubleQAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.q1_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        self.q2_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])


    # Double Q- learning :  Q1(s,a) <- Q1(s,a) + alpha * (R' + gamma * Q2 (s',arg_max_a Q1(s',a)) - Q1 (s,a))
    def learn(self, state, action, reward, next_state):
        if np.random.rand() < 0.5:
            current_q = self.q1_table[state][action]
            max_action = self.q1_table[next_state].index(max(self.q1_table[next_state])) ##2개면??
            new_q = (current_q + self.learning_rate *
                     (reward + self.discount_factor * self.q2_table[next_state][max_action] - current_q))
            self.q1_table[state][action] = new_q
        else:
            current_q = self.q2_table[state][action]
            max_action = self.q2_table[next_state].index(max(self.q2_table[next_state])) ##2개면??
            new_q = (current_q + self.learning_rate *
                     (reward + self.discount_factor * self.q1_table[next_state][max_action] - current_q))
            self.q2_table[state][action] = new_q


    # choose A from S using policy epsilon-greedy derived from Q1 + Q2
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # random action choice
            action = np.random.choice(self.actions)
        else:
            # q max action choice
            q_table = []
            for i in range(len(self.q1_table[state])):
                q_table.append(self.q1_table[state][i] + self.q2_table[state][i])
            action = self.arg_max(q_table)
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
    agent = DoubleQAgent(actions=env.possible_actions)
    step = []

    # loop for each episode
    for episode in range(1000):
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
        state_action = agent.q1_table[str(state)]
        action = agent.arg_max(state_action)
        #q_table = []
        #for i in range(len(agent.q1_table[str(state)])):
        #    q_table.append(agent.q1_table[str(state)][i] + agent.q2_table[str(state)][i])
        #action = agent.arg_max(q_table)

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
