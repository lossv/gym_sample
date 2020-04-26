from gym.envs.registration import register
register(
    id='MYGUESSNUMBER-v0',
    entry_point='gym_test.env.env_guess_number:guess_number',
)