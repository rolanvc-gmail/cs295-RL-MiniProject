import gym
import torch
from ddpg_agent import Agent

env = gym.make('BipedalWalker-v3')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)
episode = 16000
agent.actor_local.load_state_dict(torch.load('./actor/checkpoint_actor_ep{}.pth'.format(episode), map_location="cpu"))

state = env.reset()
action = agent.act(state, False)
done = False

tot_reward = 0
while not done:
    next_state, reward, done, info = env.step(action[0])
    tot_reward += reward
    env.render()
