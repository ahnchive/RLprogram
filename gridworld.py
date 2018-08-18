# gridworld.py
# by ahnchive@gmail.com, 07/2018
# this file contains several gridworld environments for RL programming
# 1. Plain Gridworld (modified the gridworld code from RLcode (https://github.com/rlcode/reinforcement-learning/blob/master/1-grid-world/1-policy-iteration/environment.py))
# 2. Windy Gridworld (modified the gridworld code from RLcode (https://github.com/rlcode/reinforcement-learning/blob/master/1-grid-world/1-policy-iteration/environment.py))

HEIGHT = 7  # grid height (#ofrows)
WIDTH = 10  # grid width (#ofcols)
TRANSITION_PROB = 1
POSSIBLE_ACTIONS = [0, 1, 2, 3]  # left, right, up, down??
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # actions in coordinates (x,y)
REWARDS = []

WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0] #wind strength for each col

START = [0, 3]
GOAL = [7, 3]

class WindyGW:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[-1] * HEIGHT for _ in range(WIDTH)] # reward -1 for all states
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[7][3] = 0  # reward 0 for the terminal state (7,3)
        self.all_state = []
        self.start = START
        self.goal = GOAL

        for x in range(WIDTH): #### check
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        next = [state[0] + action[0], state[1] + action[1] + WIND[state[0]]]
        return self.check_boundary(next)

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
                    if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
                    if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state



class GW:
    def __init__(self):
        self.transition_probability = TRANSITION_PROB
        self.width = WIDTH
        self.height = HEIGHT
        self.reward = [[-1] * WIDTH for _ in range(HEIGHT)]
        self.possible_actions = POSSIBLE_ACTIONS
        self.reward[0][0] = 0  # terminal state ; reward 0
        self.reward[HEIGHT-1][WIDTH-1] = 0  # terminal state; reward 0
        self.all_state = []

        for x in range(WIDTH): #### check
            for y in range(HEIGHT):
                state = [x, y]
                self.all_state.append(state)

    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]

    def state_after_action(self, state, action_index):
        action = ACTIONS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])

    @staticmethod
    def check_boundary(state):
        state[0] = (0 if state[0] < 0 else WIDTH - 1
                    if state[0] > WIDTH - 1 else state[0])
        state[1] = (0 if state[1] < 0 else HEIGHT - 1
                    if state[1] > HEIGHT - 1 else state[1])
        return state

    def get_transition_prob(self, state, action):
        return self.transition_probability

    def get_all_states(self):
        return self.all_state
