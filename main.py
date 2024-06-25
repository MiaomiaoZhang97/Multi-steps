import numpy as np
import torch
import gym
import argparse
import os

import utils
import TD3
import DDPG
import DDPG_N
import TD3_N


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
    parser = argparse.ArgumentParser()
    #parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="DDPG")
    parser.add_argument("--env", default="HalfCheetah-v3")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start-steps", default=1e4, type=int,
                        help='Number of steps for the warm-up stage using random policy')
    parser.add_argument("--eval-freq", default=5000, type=int, help='Number of steps per evaluation')
    parser.add_argument("--steps", default=1e6, type=int, help='Maximum number of steps')

    parser.add_argument("--discount", default=0.99, help='Discount factor')
    parser.add_argument("--la", default=0.8, type=float)
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
    parser.add_argument("--n-step", default=0, type=int)

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
    if args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

	elif args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)

    elif args.policy == "DDPG_N":
        policy = DDPG_N.DDPG_N(**kwargs)

    elif args.policy == "TD3_N":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq

        policy = TD3_N.TD3_N(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# Select action randomly or according to policy
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

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
