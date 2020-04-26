最近打算做一个关于魂斗罗的强化学习训练，于是在网上找了一些怎么搭建Gym搭建的资料，现在做一个记录

魂斗罗的gym强化学习环境我已经搭建好了,并且已经打包发布到Pypi上了，[点击这里](https://lossyou.com/post/Gym-Contra)有关项目的详细说明

整个[gym_sample](https://github.com/OuYanghaoyue/gym_sample)环境我已经上传到git，有兴趣的同学可以自己clone或者fork下来看看样例[gym_sample](https://github.com/OuYanghaoyue/gym_sample)
<!-- more -->
## 综述
Reinforcement Learning 已经经过了几十年的发展，发展壮大。近些年来，跟随着机器学习的浪潮开始发展壮大。多次战胜围棋冠军柯洁，以及在DOTA2、星际争霸等游戏中超凡表现，成为了众人追捧的明星。目前OpenAI作为世界NO.1的AI研究机构，构建的GYM，成为衡量强化学习算法的标准工具。通过OpenAI 的Gym直接构建自己的环境，从而利用目前现有的算法，直接求解模型。

包含大量自我理解，肯定存在不正确的地方，希望大家指正

## RL and GYM
RL 考虑的是agent如何在一个环境中采取行动，以最大化一些累积奖励。
其中主要包含的是2个交互：
agent对env作出动作 改变env

env 给出奖励和新的状态 给agent
其中Gym就是OpenAI所搭建的env。

具体的安装 和 介绍 主页很详细

[Gym主页](https://gym.openai.com/) 以及 [DOC](https://gym.openai.com/docs/)

简单的安装方法如下


```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```
你也可是使用


```
pip install -e .[all]
```
来安装全部的Gym现成的环境

安装好Gym之后，可以在annaconda 的 env 下的 环境名称 文件夹下 python sitpackage 下。

在调用Gym的环境的时候可以利用：

```
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
```


GYM的文件夹下 主要包含：

文件结构如下



envs 所有环境都保存在这个文件下
spaces 环境所定义的状态、动作空间
utils 环境中使用的一组常用实用程序
warppers 包装
init 读取时初始化
core 核心环境，直接链接到给定的环境
GYM 创建的环境主要在envs中，在这个里面可以找到常用的几个环境，比如说cart-pole, MountainCar等等。
自我构建的GYM环境都应该在放在envs下子文件夹中的一个py文件中的类。
例如：


```
gym\envs\classic_control\cartpole.py
```

## Gym register

所有构建的环境都需要调用GYM库，然后再通过GYM库来调用所写的环境。所以需要现在GYM的内部构件一个内链接，指向自己构建的环境。
registry 主要在

envs下 __init__.py 文件下


```
`register(`
 	`id='CartPole-v1',`
 	`entry_point='gym.envs.classic_control:CartPoleEnv',`
 	`max_episode_steps=500,`
 	`reward_threshold=475.0,`
 `)`
```


id 调用所构建的环境的名称 调用该环境的时候 所起的名字
==注：名字包含一些特殊符号的时候，会报错==

entry_point 所在的位置
例如上述： 存在gym 文件夹下 classic_control文件夹下
算法所需的参数
2 在所在文件夹下	
建立 _init_ 文件，在下面调用


```
from gym.envs.classic_control.cartpole import CartPoleEnv
```


其中是cartpole是环境所存在的文件名字，CartPoleEnv是该文件下的类。

Gym 环境构建
自我构建的环境为一个类。主要包含：变量、函数

必须的变量
这个类包含如下两个变量值：state 和 action
对应的两个空间为observation _space 和 action _space
这两个空间必须要用 space 文件夹下的类在__init__中进行定义。
其中 state是一个 object 一般为一个np.array 包含多个状态指示值。

必须存在的函数
step 利用动作 环境给出的一下步动作 和 环境给出的奖励（核心）

这个函数 承担了最重要的功能，是所构建环境所实现功能的位置

输入为 动作 输出为

    1. 下一个状态值 object
    1. 反馈 float 值
    1. done（终结标志） 布尔值 0 或者1
    1. info（对调试有用的任何信息） any
    1. reset	重置环境
    1. 将状态设置为初始状态，返回： 状态值

- render 在图形界面上作出反应
可以没有，但是必须存在

- close 关闭图形界面

- seed 随机种子
可以没有，但是必须存在

## 状态、动作空间的构建
主要分为离散空间和连续空间：
连续空间主要由spaces.Box定义，例如：


```
self.action_space = spaces.Box(low=-10, high=10, shape=(1,2))
```


上面定义了一个取值范围在（-10，10）的变量 维度为1，2

离散空间主要有

spaces.Discrete，例如


```
self.observation_space = spaces.Discrete(2)
```

上面定义了一个变量空间范围为[0,2) 之间的整数

spaces.MultiBinary， 例如

```
self.observation_space = spaces.MultiBinary(2)
```

上面定义了一个变量空间为0，1的2维整数变量

spaces.MultiBinary， 例如


```
self.observation_space = MultiDiscrete（）
```

其他还可以定义一个元组或者字典 等变量空间。
# 下面仔细说明一下Gym的文结构

## Gym的文件结构

```
├── gym_test
│   ├── env
│   │   ├── env_guess_number.py
│   │   ├── __init__.py
│   └── __init__.py


```

在__init__.py文件下你要包含如下代码

```
from gym.envs.registration import register
register(
    id='MYGUESSNUMBER-v0',
    entry_point='gym_test.env.env_guess_number:guess_number',
)
# gym_test.env是相对于项目名字的gym的路径  
# env_guess_number是env_guess_number.py
# guess_number 是类名
```

## 如何使用自定义的Gym 环境？只需要这样

```
import gym
import gym_test.env
env = gym.make('MYGUESSNUMBER-v0')
```

您必须在PYTHONPATH中安装gym_sample目录或从父目录来使用您自定义的gym环境。

## 例如


```
整个项目结构：
├── gym_test
│   ├── env
│   │   ├── env.guess_number.py
│   │   ├── __init__.py
│   │   └── __pycache__
│   │       └── __init__.cpython-37.pyc
│   └── __init__.py
├── README.md
└── test.py

```

-------------------
__init__.py 文件：
-------------------

```
from gym.envs.registration import register
register(
    id='MYGUESSNUMBER-v0',
    entry_point='gym_test.env.env_guess_number:guess_number',
)
```

-------------------
env_guess_number.py文件：
-------------------

```
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

```
-------------------
## test.py文件


```
import gym

import gym_test.env

env = gym.make('MYGUESSNUMBER-v0')

obs = env.reset()

for step in range(10000):
    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
```
