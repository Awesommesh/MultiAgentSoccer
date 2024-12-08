import ray
from ray import tune
import pickle
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls
import gymnasium as gym
import numpy as np
import os
import soccer_fours, soccer_threes, soccer_twos
import argparse
from utils import create_rllib_env

parser = argparse.ArgumentParser(description="Train multi-agent Soccer")

# Add arguments
parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
parser.add_argument('-n', '--num_agents_per_team', type=int, choices=[2, 3, 4], 
    default=2, help='number of players per team in the soccer environmnet (must be either 2, 3, or 4)')
parser.add_argument('--timescale', type=int, default=1, help='timescale for environment')
parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to test')
parser.add_argument('--ckpt_path1', type=str, required=True, help='Path to blue team checkpoint')
parser.add_argument('--ckpt_path2', type=str, required=True, help='Path to red team checkpoint')
parser.add_argument('--dont_render', action='store_true', help='dont render simulation')

# Parse the arguments
args = parser.parse_args()
num_per_team = args.num_agents_per_team
time_scale = args.timescale
render = True if not args.dont_render else False

# Initialize Ray
ray.init()
blue_policy = "random"
purple_policy = "random"
if args.ckpt_path1 != "random":
    ckpt_p = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.ckpt_path1,
    )
    param_pkl = os.path.join(ckpt_p, "algorithm_state.pkl")
    print(param_pkl)
    with open(param_pkl, "rb") as f:
        config = pickle.load(f)

    print("Loaded algorithm state!")
    # no need for parallelism on evaluation
    config["num_workers"] = 0
    config["num_gpus"] = 0
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 3])
    config["observation_space"] = observation_space
    config["action_space"] = action_space
    # create a dummy env since it's required but we only care about the policy
    tune.registry.register_env("Soccer", create_rllib_env)
    print(config.keys(), config)
    cls = get_trainable_cls("PPO")
    print(cls)
    agent = cls(config=config)
    # load state from checkpoint
    agent.restore(ckpt_p)
    # get policy for evaluation
    blue_policy = agent.get_policy("default")
    state = blue_policy.get_initial_state()
if args.ckpt_path2 != "random":
    ckpt_p = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        args.ckpt_path2,
    )
    param_pkl = os.path.join(ckpt_p, "algorithm_state.pkl")
    print(param_pkl)
    with open(param_pkl, "rb") as f:
        config = pickle.load(f)

    print("Loaded algorithm state!")
    # no need for parallelism on evaluation
    config["num_workers"] = 0
    config["num_gpus"] = 0
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 3])
    config["observation_space"] = observation_space
    config["action_space"] = action_space
    # create a dummy env since it's required but we only care about the policy
    tune.registry.register_env("Soccer", create_rllib_env)
    print(config.keys(), config)
    cls = get_trainable_cls("PPO")
    print(cls)
    agent = cls(config=config)
    # load state from checkpoint
    agent.restore(ckpt_p)
    # get policy for evaluation
    purple_policy = agent.get_policy("default")
'''
print("Testing retrieval")
# Gets best trial based on max accuracy across all training iterations.
best_trial = agent.get_best_trial("env_runners/episode_reward_mean", mode="max")
print(best_trial)
# Gets best checkpoint for trial based on accuracy.
best_checkpoint = agent.get_best_checkpoint(
    trial=best_trial, metric="env_runners/episode_reward_mean", mode="max"
)
print(best_checkpoint)'''

#Create env
if num_per_team == 4:
    env = soccer_fours.make(time_scale=time_scale, render=render)
elif num_per_team == 3:
    env = soccer_threes.make(time_scale=time_scale, render=render)
elif num_per_team == 2:
    env = soccer_twos.make(time_scale=time_scale, render=render)
else:
    print(f"GOT UNEXPECTED NUMBER OF PLAYERS PER TEAM! {num_per_team}")
print("Observation Space: ", env.observation_space.shape)
print("Action Space: ", env.action_space.shape)

team0_reward = 0
team1_reward = 0
obs = env.reset()
team0_goals, team1_goals = 0, 0
draws = 0
num_ep = 0
while True and num_ep < args.num_episodes:
    actions = {}
    for i in range(num_per_team):
        if blue_policy != "random":
            actions[i], state, info = blue_policy.compute_single_action(obs[i], state=state)
            
        else:
            actions[i] = env.action_space.sample()
    for i in range(num_per_team):
        if purple_policy != "random":
            actions[num_per_team+i] = purple_policy.compute_single_action(obs[num_per_team+i])[0]
        else:
            actions[num_per_team+i] = env.action_space.sample()
    obs, reward, done, info = env.step(actions)
    for i in range(num_per_team):
        team0_reward += reward[i]
        team1_reward += reward[num_per_team+i]
    if done["__all__"]:
        num_ep += 1
        print("Total Reward: ", team0_reward, " x ", team1_reward)
        team0_goals += 1 if team0_reward > 0 else 0
        team1_goals += 1 if team1_reward > 0 else 0
        draws += 1 if team0_reward == 0.0 else 0
        print(f"Goals scored: {team0_goals} x {team1_goals}. {draws} Draws")
        team0_reward = 0
        team1_reward = 0
        obs = env.reset()
env.close()