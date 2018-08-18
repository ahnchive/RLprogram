# by ahnchive@gmail.com, 07/2018
# code modified from the ‘mc_agent.py’ by RLcode team (https://github.com/rlcode)


import numpy as np
import random
from collections import defaultdict
from gridworld import WindyGW
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

''' (incremental) every-visit MC control algorithm (modified from Sutton & Barto, 2018)

Loop for each episode
    Generate an episode using policy derived from Q (epsilon-greedy)
    G <- 0
    Loop for each step of episode, in a reversed order
        G <- gamma * G + R'
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a)) 


cf. original one
Loop for each episode
    Generate an episode following pi
    G <- 0
    Loop for each step of episode, in a reversed order
        G <- gamma * G + R'
        Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a)) 
        A* <- argmax_a Q(s,a)
        for all a in A:
            pi(a|s) <- 1 -epsilon -epsilon/|A|, if a = A*
                    <- epsilon/|A|, if a =! A*

'''


class MCagent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.samples = []
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # Generate an episode using policy derived from Q (epsilon-greedy)
    def gen_episode(self, env):

        time = 0

        # initialize S
        state = env.start

        # loop for each step of episode
        while time < 1000:

            time += 1

            # choose A from S using policy derived from Q (epsilon-greedy)
            action = self.get_action(str(state))
            # take action A, observe R', S'
            reward = env.get_reward(state, action)
            next_state = env.state_after_action(state, action)
            # save samples : [[state,action, reward],..]
            self.samples.append([state, action, reward])

            state = next_state

            # until S is terminal
            if state == env.goal:
                break

        #print(time)
        return time


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

    # MC incremental learning Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a))
    def learn(self, state, action, G_t):
        current_q = self.q_table[state][action]
        new_q = (current_q + self.learning_rate *(G_t - current_q))
        self.q_table[state][action] = new_q

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
    agent = MCagent(actions=env.possible_actions)
    step = []

    # loop for each episode
    for episode in range(5000):
        print("episode %d"%episode, end='\t:')

        # generate an episode using policy derived from Q (epsilon-greedy)
        agent.samples.clear()
        time = agent.gen_episode(env)

        print(time)
        step.append(time)

        #loop for each step of episode, in a reversed order
        G_t = 0
        for sample in reversed(agent.samples):

            state = str(sample[0])
            action = sample[1]
            reward = sample[2]

            G_t = agent.discount_factor * (reward + G_t)
            agent.learn(state,action,G_t)




    # draw the number of steps per episode
    plt.figure()
    plt.plot(step)
    plt.xlabel('episode')
    plt.ylabel('num of steps')
    #plt.show()

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

        if action == 0:
            print("L", end='')
        elif action == 1:
            print("R", end=''),
        elif action == 2:
            print("U", end=''),
        elif action == 3:
            print("D", end=''),

        if state == env.goal:
            print("G")
            break
