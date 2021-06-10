import numpy as np
import torch
import gym
import argparse
import os
import datetime
import utils
import DDPG_a

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    print(f"Starting at:{starttime}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG_a")  # Policy name (TD3, DDPG or OurDDPG)
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

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

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
              # "policy_freq": args.policy_freq
              }


    # Initialize policy
    # Target policy smoothing is scaled wrt the action scale
    # Step 1: Initialize critic and actor networks
    # Step 2: Initialize target networks
    policy = DDPG_a.DDPG_a(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        # default policy filename is: f"{args.policy}_{args.env}_{args.seed}"
        policy.load(f"./models/{policy_file}")

    #Step 3: Initialize Replay Buffer
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    # Step 4: for t=1 to T do:
    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # Step 5: Select Action wih exploration noise
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Step 6: Store transition tuple (s, a, r, s) in replay buffer.  Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        # Steps 7- seems to be inside of policy.train().
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        # evaluation is based on number of time steps.
        # Let's change this and re-rerun.

        # let's do this only when it's a done...
            if (episode_num + 1) % 10 == 0:
                evaluations.append(eval_policy(policy, args.env, args.seed))
                np.save(f"./results/{file_name}_commented", evaluations)
                # ./results/filename contains a list of evaluations every 10 episodes.
                # leter to plot, we just create a list of multiples of 10...
                if args.save_model and (episode_num + 1) % 100:
                    policy.save(f"./models/{file_name}")

    endtime = datetime.datetime.now()
    timediff = endtime - starttime
    print(f"Ending at:{endtime}")
    print(f"Difference: {timediff}")
