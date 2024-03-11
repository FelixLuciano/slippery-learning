import random
import sys

import numpy as np
import pandas as pd

class Sarsa:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explora o espaço de ações
        return np.argmax(self.q_table[state]) # Faz uso da tabela Q

    def train(self):
        rewards_per_episode = []
        actions_per_episode = []
        for i in range(1, self.episodes+1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0
            action = self.select_action(state=state)
            while not done:
                next_state, reward, done, _, _ = self.env.step(action)
                next_action = self.select_action(state=next_state)
                rewards=rewards+reward
                self.q_table[state][action] = self.q_table[state][action] + self.alpha*(reward+self.gamma*self.q_table[next_state][next_action]-self.q_table[state][action])
                actions=actions+1
                state = next_state
                action = next_action
            rewards_per_episode.append(rewards)
            actions_per_episode.append(actions)
            if i % 100 == 0:
                sys.stdout.write("Episodes: " + str(i) +'\r')
                sys.stdout.flush()
            
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        data_rewards = pd.DataFrame({'Episodes': range(1, len(rewards_per_episode)+1), 'Rewards': rewards_per_episode})
        data_actions = pd.DataFrame({'Episodes': range(1, len(actions_per_episode)+1), 'Actions': actions_per_episode})
        return self.q_table, data_rewards, data_actions
