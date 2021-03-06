import gym
from gym.wrappers import Monitor
import torch
import numpy as np
import TD3
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="DDPG")  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="BipedalWalker-v3")  # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)  # Discount factor
parser.add_argument("--tau", default=0.005)  # Target network update rate
parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

file_name = f"{args.policy}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")

env = gym.make(args.env)

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {"state_dim": state_dim,
          "action_dim": action_dim,
          "max_action": max_action,
          "discount": args.discount,
          "tau": args.tau,
          "policy_noise": args.policy_noise * max_action,
          "noise_clip": args.noise_clip * max_action,
          "policy_freq": args.policy_freq}

# Initialize p
# Target policy smoothing is scaled wrt the action scale
# Step 1: Initialize critic and actor networks
# Step 2: Initialize target networks
policy = TD3.TD3(**kwargs)
policy.load(f"./models/{file_name}")



def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            eval_env.render()
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

if __name__== "__main__":
    ave_reward = eval_policy(policy, args.env, args.seed)
    print(ave_reward)