import gym
import torch
import numpy as np
from ddpg_agent import Agent
import matplotlib.pyplot as plt
import pickle
from collections import deque
import datetime
gym.logger.set_level(40)
env = gym.make('BipedalWalker-v3')

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# MAX_EPISODES = 20000
MAX_EPISODES = 50000
MAX_REWARD = 300
MAX_STEPS = 2000  # env._max_episode_steps
MEAN_EVERY = 100

start_episode = 0
# Step 1. Randomly initialize critic and actor networks.
# Step 2. Initialize target networks.
# Step 3. Initialize replay buffer. This is Agent's self.memory initialized in __init__().
# Agent is  ddpg_agent.Agent. Agent maintains a local actor and critic, as well as target Actors and Critics.
# Actors and Critics are model.Actor and model.Critic respectively.
#Step 5. Initialize a random process N for action exploration.
agent = Agent(state_size=state_dim, action_size=action_dim, random_seed=0)
LOAD = False
noise = 1

if LOAD:
    start_episode = 10000
    agent.actor_local.load_state_dict(torch.load('./actor/checkpoint_actor_ep10000.pth', map_location="cpu"))
    agent.critic_local.load_state_dict(torch.load('./critic/checkpoint_critic_ep10000.pth', map_location="cpu"))
    agent.actor_target.load_state_dict(torch.load('./actor/checkpoint_actor_t_ep10000.pth', map_location="cpu"))
    agent.critic_target.load_state_dict(torch.load('./critic/checkpoint_critic_t_ep10000.pth', map_location="cpu"))

start_time = datetime.datetime.now()
print("Start time is {}".format(start_time.strftime("%H:%M:%S")))
scores = []
mean_scores = []
# deques are list-like containers with fast appends and pops on either end.
last_scores = deque(maxlen=MEAN_EVERY)
distances = []
mean_distances = []
last_distance = deque(maxlen=MEAN_EVERY)
losses_mean_episode = []
rewards = []
# step 4: For episode=1, M, do:
for ep in range(start_episode + 1, MAX_EPISODES + 1):
    # Step 6: Receive initial observation state
    state = env.reset()
    total_reward = 0
    total_distance = 0
    actor_losses = []
    critic_losses = []

    # Step 7: for t =1,T do:
    for t in range(MAX_STEPS):

        # env.render()

        # Step 8: Select action according to the current policy and exploration noise.
        action = agent.act(state, noise)

        # Step 9: Execute action and observe reward, and observe new state.
        next_state, reward, done, info = env.step(action[0])

        # Step 10: Store transition (s, a, r, s) in R
        # Step 11: Sample a random minibatch of N transitions (s, a, r, s) from R
        # Step 12: Set y = ri + gQ(s, u |theta)
        # Step 13: Update critic
        # Step 14: Update actor
        # Step 15: UPdate target network
        # agent.step() does all the steps above
        actor_loss, critic_loss = agent.step(state, action, reward, next_state, done)
        if actor_loss is not None:
            actor_losses.append(actor_loss)
        if critic_loss is not None:
            critic_losses.append(critic_loss)
        state = next_state.squeeze()
        state = next_state
        total_reward += reward
        if reward != -100:
            total_distance += reward
        if done:
            break

    if len(actor_losses) >= 1 and len(critic_losses) >= 1:
        mean_loss_actor = np.mean(actor_losses)
        mean_loss_critic = np.mean(critic_losses)
        losses_mean_episode.append((ep, mean_loss_actor, mean_loss_critic))
    else:
        mean_loss_actor = None
        mean_loss_critic = None

    print(
        '\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tactor_loss: {},\tcritic_loss:{}'.format(ep, MAX_EPISODES,
                                                                                                       total_reward,
                                                                                                       total_distance,
                                                                                                       mean_loss_actor,
                                                                                                       mean_loss_critic),
        end="")
    rewards.append(total_reward)

    scores.append(total_reward)
    distances.append(total_distance)
    last_scores.append(total_reward)
    last_distance.append(total_distance)
    mean_score = np.mean(last_scores)
    mean_distance = np.mean(last_distance)
    FILE = 'record.dat'
    data = [ep, total_reward, total_distance, mean_loss_actor, mean_loss_critic]
    with open(FILE, "ab") as f:
        pickle.dump(data, f)

    if mean_score >= 300:
        print('Task Solved')
        torch.save(agent.actor_local.state_dict(), './actor/checkpoint_actor_best_ep' + str(ep) + '.pth')
        torch.save(agent.critic_local.state_dict(), './critic/checkpoint_critic_best_ep' + str(ep) + '.pth')
        torch.save(agent.actor_target.state_dict(), './actor/checkpoint_actor_best_t_ep' + str(ep) + '.pth')
        torch.save(agent.critic_target.state_dict(), './critic/checkpoint_critic_best_t_ep' + str(ep) + '.pth')
        break

    if ((ep % MEAN_EVERY) == 0):
        torch.save(agent.actor_local.state_dict(), './actor/checkpoint_actor_ep' + str(ep) + '.pth')
        torch.save(agent.critic_local.state_dict(), './critic/checkpoint_critic_ep' + str(ep) + '.pth')
        torch.save(agent.actor_target.state_dict(), './actor/checkpoint_actor_t_ep' + str(ep) + '.pth')
        torch.save(agent.critic_target.state_dict(), './critic/checkpoint_critic_t_ep' + str(ep) + '.pth')
        mean_scores.append(mean_score)
        mean_distances.append(mean_distance)
        print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tactor_loss: {},\tcritic_loss:{}'.format(
            ep, MAX_EPISODES,
            mean_score,
            mean_distance, mean_loss_actor,
            mean_loss_critic))
        FILE = 'record_mean.dat'
        data = [ep, mean_score, mean_distance, mean_loss_actor, mean_loss_critic]
        with open(FILE, "ab") as f:
            pickle.dump(data, f)
env.close()
with open("rewards.dat", "wb") as rewards_file:
    pickle.dump(rewards, rewards_file)
end_time = datetime.datetime.now()
time_diff = end_time - start_time


def days_hours_minutes(td):
    return td.days, td.seconds//3600, (td.seconds//60)%60


days, hrs, minutes = days_hours_minutes(time_diff)

print("End time is {}".format(end_time.strftime("%H:%M:%S")))
print("Run time for {} episodes is:{} days, {} hours, and {} mins ".format(MAX_EPISODES, days, hrs, minutes))
