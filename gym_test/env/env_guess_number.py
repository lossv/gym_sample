import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class guess_number(gym.Env):
    """Hotter Colder
    The goal of hotter colder is to guess closer to a randomly selected guess_number

    After each step the agent receives an observation of:
    0 - No guess yet submitted (only after reset)
    1 - Guess is lower than the target
    2 - Guess is equal to the target
    3 - Guess is higher than the target

    The rewards is calculated as:
    (min(action, self.guess_number) + self.range) / (max(action, self.guess_number) + self.range)

    Ideally an agent will be able to recognise the 'scent' of a higher reward and
    increase the rate in which is guesses in that direction until the reward reaches
    its maximum
    """

    def __init__(self):
        self.range = 1000  # +/- value the randomly select guess_number can be between
        self.bounds = 2000  # Action space bounds

        self.action_space = spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = spaces.Discrete(4)

        self.guess_number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action)

        if action < self.guess_number:
            self.observation = 1

        elif action == self.guess_number:
            self.observation = 2

        elif action > self.guess_number:
            self.observation = 3

        reward = ((min(action, self.guess_number) + self.bounds) / (max(action, self.guess_number) + self.bounds)) ** 2

        self.guess_count += 1
        done = self.guess_count >= self.guess_max

        return self.observation, reward, done, {"guess_number": self.guess_number, "guesses": self.guess_count}

    def reset(self):
        self.guess_number = self.np_random.uniform(-self.range, self.range)
        print('guess number = ', self.guess_number)
        self.guess_count = 0
        self.observation = 0
        return self.observation
