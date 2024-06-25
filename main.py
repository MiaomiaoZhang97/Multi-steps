from collections import deque
import utils
import numpy as np
import gym
import numpy as np
import torch
import gym
import argparse
import os
import utils
import random
import TD3_N


def eval_policy(policy, env_name, seed, eval_episodes=10, eval_cnt=None):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward  #算法的评估


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="TD3_N")
    parser.add_argument("--env", default="HalfCheetah-v3")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start-steps", default=1e4, type=int,
                        help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval-freq", default=5000, type=int, help='Number of steps per evaluation')
    parser.add_argument("--steps", default=1e6, type=int, help='Maximum number of steps')

    parser.add_argument("--discount", default=0.99, help='Discount factor')
    parser.add_argument("--lambda", default=0.8, type=float)
    parser.add_argument("--tau", default=0.005, type=float, help='Target network update rate')

    parser.add_argument("--alpha", default=0.005, type=float)

    parser.add_argument("--actor-lr", default=1e-3, type=float)
    parser.add_argument("--critic-lr", default=1e-3, type=float)
    parser.add_argument("--hidden-sizes", default='256,256', type=str)
    parser.add_argument("--batch-size", default=256, type=int)  # Batch size for both actor and critic

    parser.add_argument("--save-model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load-model", default="")  # Model load file name, "" doesn't load, "default" uses file_name

    parser.add_argument("--expl-noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--policy-noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", default=0.5, type=float)  # Range to clip target policy noise

    parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--N-step", default=2, type=int)

    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")


    env = gym.make(args.env)

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "hidden_sizes": [int(hs) for hs in args.hidden_sizes.split(',')],
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "device": device,
        "alpha": args.alpha,
    }
	
    if args.policy == "TD3_N":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq

        policy = TD3_N.TD3_N(**kwargs)

    if args.load_model != "":
        policy.load("./models/{}".format(args.load_model))

    #Initialize replay buffer with state and action dimension & buffer size
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    # Initialize temporary buffer D'
    exp_buffer = deque()

    eval_cnt = 0
    eval_return = [eval_policy(policy, args.env, args.seed, eval_cnt=eval_cnt)]
    eval_cnt += 1

    # Initialize environment
    state, done = env.reset(), False

    # Initialize episode timesteps
    episode_timesteps = 0
    episode_reward = 0
    episode_num = 0

    # Start training steps until max timesteps
    for t in range(int(args.steps)):
        episode_timesteps += 1

        # select action randomly or according to policy
        if t < args.start_steps:
            action = env.action_space.sample()
        else:
            action = (
                    policy.select_action(np.array(state))
                    + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        exp_buffer.append((state, action, next_state, reward, done_bool))

        state = next_state
        episode_reward += reward

        if t >= args.start_steps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            len_buffer = len(exp_buffer)
            d_reward = 0
            for i in range(len_buffer):
                discounted_reward = 0
                gamma = args.lambda

                state, action, next_state, _, done_bool = exp_buffer[i]
                for j in range(i, min(i + args.N_step, len_buffer)):
                    _, _, _, r, _ = exp_buffer[j]
                    discounted_reward += r * gamma
                    gamma *= args.la
                ds_factor = (1 - args.lambda) / (args.la * (1 - gamma / args.lambda))
                discounted_reward = ds_factor * discounted_reward
                d_reward += discounted_reward
                replay_buffer.add(state, action, next_state, discounted_reward, done_bool)
            exp_buffer.clear()
            print(
                    "Total T: {} Episode Num: {} Episode T: {} discounted_Reward: {} Reward: {}".format(t + 1,
                                                   episode_num + 1, len_buffer, d_reward, episode_reward))

            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t+1) % args.eval_freq == 0:
            eval_return.append(eval_policy(policy, args.env, args.seed,  eval_cnt=eval_cnt))
            np.save(f"./results/{file_name}", eval_return)
            eval_cnt += 1
