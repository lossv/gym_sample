import gym

import gym_test.env

env = gym.make('MYGUESSNUMBER-v0')

obs = env.reset()

for step in range(10000):
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)