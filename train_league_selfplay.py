import ray
from ray import tune
import argparse
import gymnasium as gym
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from leagueselfplaycallback import LeagueBasedSelfPlayCallback
from utils import create_rllib_env
from ray.tune.registry import get_trainable_cls
import os
import pickle

# Model config settings: https://docs.ray.io/en/latest/rllib/rllib-models.html#rnns
# Starter ray code reference: https://github.com/bryanoliveira/soccer-twos-starter/tree/main

def train_ppo(args):
    #Setup environment
    num_per_team = args.num_agents_per_team
    time_scale = args.timescale
    train_bs = args.train_bs
    num_workers = args.num_workers
    num_envs_per_worker = args.num_envs_per_worker
    num_epochs = args.num_epochs
    default_policy_weights = None

    # Initialize Ray
    ray.init()
    print("initialized ray")

    if args.ckpt is not None:
        # Load the checkpoint to access the 'default' policy weights
        checkpoint_path = ckpt_p = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            args.ckpt,
        ) 
        param_pkl = os.path.join(checkpoint_path, "algorithm_state.pkl")
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
        agent.restore(checkpoint_path)
        # get policy for evaluation
        policy = agent.get_policy("default")
        default_policy_weights = policy.get_weights()

    class SelfPlayUpdateCallback(DefaultCallbacks):
        def on_algorithm_init(self, **info):
            print("---- Setting same opponents!!! ----")
            trainer = info["algorithm"]
            if args.ckpt:
                print("---- Setting from ckpt opponents!!! ----")
                trainer.set_weights(
                    {
                        "default": default_policy_weights
                    }
                )
            else:
                trainer.set_weights(
                    {
                        "opponent_3": trainer.get_policy("opponent_1").get_weights(),
                        "opponent_2": trainer.get_policy("opponent_1").get_weights(),
                    }
                )

        def on_train_result(self, **info):
            """
            Update multiagent oponent weights when reward is high enough
            """
            main_rew = info["result"]["env_runners"]["hist_stats"].pop("policy_default_reward")
            won = 0
            total = 0
            for i in range(len(main_rew)-1, -1, -num_per_team):
                total += 1
                if main_rew[i] > 0:
                    won += 1
            win_rate = won / total
            print("win rate:", win_rate)
            info["result"]["env_runners"]["win_rate"] = win_rate
            trainer = info["algorithm"]
            if win_rate > 0.8:
                print("---- Updating opponents!!! ----")
                
                trainer.set_weights(
                    {
                        "opponent_3": trainer.get_policy("opponent_2").get_weights(),
                        "opponent_2": trainer.get_policy("opponent_1").get_weights(),
                        "opponent_1": trainer.get_policy("default").get_weights(),
                    }
                )


    #hardcoded all envs observation and action space
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([3, 3, 3])

    tune.registry.register_env("Soccer", create_rllib_env)

    env_config = {
        "time_scale": time_scale,
        "num_per_team": num_per_team,
        "num_workers": num_workers,
        "num_envs_per_worker": num_envs_per_worker,
        "render": False,
    }
    team_policy_map = {"blue": "default"}
    def policy_mapping_fn(agent_id, *args, **kwargs):
        #ep_obj = args[0] https://github.com/ray-project/ray/blob/master/rllib/evaluation/episode_v2.py
        '''if agent_id < num_per_team:
            return "default"  # Choose 01 policy for agent_01
        else:
            return np.random.choice(
                ["opponent_1", "opponent_2", "opponent_3"],
                size=1,
                p=[0.5, 0.25, 0.25],
            )[0]'''
        team_id = "blue" if agent_id < num_per_team else "red"
        if team_id not in team_policy_map:
            # Randomly select a policy for this team (ensure this list contains your available policies)
            team_policy_map[team_id] = np.random.choice(
                ["opponent_1", "opponent_2", "opponent_3"],
                size=1,
                p=[0.5, 0.25, 0.25],
            )[0]
        policy = team_policy_map[team_id]
        if agent_id == num_per_team*2-1:
            del team_policy_map[team_id]
        return policy


    ppo = tune.run(
        "PPO",
        name="PPO_selfplay_rec",
        config={
            # system settings
            #"num_gpus": 1,
            "num_workers": num_workers,
            "num_envs_per_env_runner": num_envs_per_worker,
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": SelfPlayUpdateCallback,
            # RL setup
            "multiagent": {
                "policies": {
                    "default": (None, observation_space, action_space, {}),
                    "opponent_1": (None, observation_space, action_space, {}),
                    "opponent_2": (None, observation_space, action_space, {}),
                    "opponent_3": (None, observation_space, action_space, {}),
                },
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": ["default"],
            },
            "env": "Soccer",
            "env_config": env_config,
            "model": {
                "vf_share_layers": False,
                "fcnet_hiddens": [512, 512],
                "fcnet_activation": "relu",
                "use_lstm": args.use_lstm,
                "max_seq_len": args.max_seq_len,
                "lstm_cell_size": args.lstm_cell_size,  # Size of the LSTM cell
                "lstm_use_prev_action": False if args.lstm_dont_use_prev_action else True,  # Whether to use previous actions and rewards as inputs
            },
            "gamma": args.gamma,
            "lr": args.lr,
            "clip_param": args.clip_param,
            "entropy_coeff": args.entropy_coeff,
            "vf_loss_coeff": args.vf_coeff,
            "lambda_": args.lambda_,
            "minibatch_size": args.sgd_minibatch_size,
            "num_epochs": args.num_sgd_iter,
            "train_batch_size": train_bs,
            "batch_mode": "complete_episodes",
        },
        stop={"timesteps_total": 30000000, "time_total_s": 43200,},  # 12h
        checkpoint_freq=20,
        checkpoint_at_end=True,
        keep_checkpoints_num=100,
        storage_path="~/repositories/MultiAgentSoccer/ray_results/",
        #restore="./ray_results/PPO_selfplay_rec/PPO_Soccer_ac781_00000_0_2024-11-05_04-02-24/checkpoint_000030",
    )
    
    # Gets best trial based on max accuracy across all training iterations.
    best_trial = ppo.get_best_trial("env_runners/episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = ppo.get_best_checkpoint(
        trial=best_trial, metric="env_runners/episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")
    

#For now only trains using PPO with self-play
def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Train multi-agent Soccer")

    # Add arguments
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('-n', '--num_agents_per_team', type=int, choices=[2, 3, 4], 
        default=2, help='number of players per team in the soccer environmnet (must be either 2, 3, or 4)')
    parser.add_argument('--timescale', type=int, default=20, help='timescale for environment')
    parser.add_argument('--train_bs', type=int, default=8000, help='batch size for training PPO')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers to instantiate for efficient sampling and rollouts')
    parser.add_argument('--num_envs_per_worker', type=int, default=1, help='number of envs to per worked instantiated for efficient sampling and rollouts')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of PPO training rounds')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma discount factor')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--clip_param', type=float, default=0.3, help='PPO clip parameter')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy_coefficient')
    parser.add_argument('--vf_coeff', type=float, default=1.0, help='vf coefficient')
    parser.add_argument('--lambda_', type=float, default=0.95, help='lambda')
    parser.add_argument('--sgd_minibatch_size', type=int, default=512, help='sgd minibatch size')
    parser.add_argument('--num_sgd_iter', type=int, default=30, help='number of sgd iterations')
    parser.add_argument('--use_lstm', action='store_true', help='Use LSTM policy model')
    parser.add_argument('--max_seq_len', type=int, default=20, help='LSTM max sequence length')
    parser.add_argument('--lstm_cell_size', type=int, default=512, help='LSTM cell size')
    parser.add_argument('--lstm_dont_use_prev_action', action='store_false', help="Don't use use previous actions and rewards as inputs to LSTM")
    parser.add_argument('--lstm_num_layers', type=int, default=3, help='LSTM number of layers')
    parser.add_argument('--self_play_freeze_freq', type=int, default=10, help='Frequency with which to update self_play policy')
    parser.add_argument('--ckpt', type=str, default=None, help="Restore from checkpoint")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.verbose:
        print(f"Using env Soccer{args.num_agents_per_team}v{args.num_agents_per_team}")
        print(f"Env timescale {args.timescale}")

    #Train agent
    train_ppo(args)

if __name__ == "__main__":
    main()

import functools

import numpy as np

import ray
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.utils import try_import_pyspiel, try_import_open_spiel
from ray.rllib.env.wrappers.open_spiel import OpenSpielEnv
from ray.rllib.examples.multi_agent.utils import (
    ask_user_for_action,
    SelfPlayLeagueBasedCallback,
    SelfPlayLeagueBasedCallbackOldAPIStack,
)
from ray.rllib.examples._old_api_stack.policy.random_policy import RandomPolicy
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

open_spiel = try_import_open_spiel(error=True)
pyspiel = try_import_pyspiel(error=True)

# Import after try_import_open_spiel, so we can error out with hints
from open_spiel.python.rl_environment import Environment  # noqa: E402


parser = add_rllib_example_script_args(default_timesteps=2000000)
parser.set_defaults(env="markov_soccer")
parser.add_argument(
    "--win-rate-threshold",
    type=float,
    default=0.85,
    help="Win-rate at which we setup another opponent by freezing the "
    "current main policy and playing against a uniform distribution "
    "of previously frozen 'main's from here on.",
)
parser.add_argument(
    "--min-league-size",
    type=float,
    default=8,
    help="Minimum number of policies/RLModules to consider the test passed. "
    "The initial league size is 2: `main` and `random`. "
    "`--min-league-size=3` thus means that one new policy/RLModule has been "
    "added so far (b/c the `main` one has reached the `--win-rate-threshold "
    "against the `random` Policy/RLModule).",
)
parser.add_argument(
    "--num-episodes-human-play",
    type=int,
    default=0,
    help="How many episodes to play against the user on the command "
    "line after training has finished.",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Algorithm state.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
        # At first, only have main play against the random main exploiter.
        return "main" if hash(episode.id_) % 2 == agent_id else "main_exploiter_0"

    def _get_multi_agent():
        names = {
            # Our main policy, we'd like to optimize.
            "main",
            # First frozen version of main (after we reach n% win-rate).
            "main_0",
            # Initial main exploiters (one random, one trainable).
            "main_exploiter_0",
            "main_exploiter_1",
            # Initial league exploiters (one random, one trainable).
            "league_exploiter_0",
            "league_exploiter_1",
        }
        policies = names
        spec = {
            mid: RLModuleSpec(
                module_class=(
                    RandomRLModule
                    if mid in ["main_exploiter_0", "league_exploiter_0"]
                    else None
                )
            )
            for mid in names
        }
        return {"policies": policies, "spec": spec}

    config = (
        get_trainable_cls("PPO")
        .get_default_config()
        # Use new API stack ...
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Soccer")
        .framework("torch")
        # Set up the main piece in this experiment: The league-bases self-play
        # callback, which controls adding new policies/Modules to the league and
        # properly matching the different policies in the league with each other.
        .callbacks(
            functools.partial(
                SelfPlayLeagueBasedCallback,
                win_rate_threshold=0.9,
            )
        )
        .env_runners(
            num_env_runners=(num_workers),
            num_envs_per_env_runner=1,
        )
        #.resources(
        #    num_cpus_for_main_process=1,
        #)
        .training(
            num_epochs=num_epochs,
        )
        .multi_agent(
            # Initial policy map: All PPO. This will be expanded
            # to more policy snapshots. This is done in the
            # custom callback defined above (`LeagueBasedSelfPlayCallback`).
            policies=_get_multi_agent()["policies"],
            policy_mapping_fn=(
                agent_to_module_mapping_fn
            ),
            # At first, only train main_0 (until good enough to win against
            # random).
            policies_to_train=["main"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs=_get_multi_agent()["spec"]
            ),
        )
    )

    # Run everything as configured.
    # Train the "main" policy to play really well using self-play.
    results = None
    if not args.from_checkpoint:
        stop = {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: args.stop_timesteps,
            TRAINING_ITERATION: args.stop_iters,
            "league_size": args.min_league_size,
        }
        results = run_rllib_example_script_experiment(config, args, stop=stop)

    # Restore trained Algorithm (set to non-explore behavior) and play against
    # human on command line.
    if args.num_episodes_human_play > 0:
        num_episodes = 0
        # Switch off exploration for better inference performance.
        config.explore = False
        algo = config.build()
        if args.from_checkpoint:
            algo.restore(args.from_checkpoint)
        else:
            checkpoint = results.get_best_result().checkpoint
            if not checkpoint:
                raise ValueError("No last checkpoint found in results!")
            algo.restore(checkpoint)

        # Play from the command line against the trained agent
        # in an actual (non-RLlib-wrapped) open-spiel env.
        human_player = 1
        env = Environment(args.env)

        while num_episodes < args.num_episodes_human_play:
            print("You play as {}".format("o" if human_player else "x"))
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == human_player:
                    action = ask_user_for_action(time_step)
                else:
                    obs = np.array(time_step.observations["info_state"][player_id])
                    if config.enable_env_runner_and_connector_v2:
                        action = algo.env_runner.module.forward_inference({"obs": obs})
                    else:
                        action = algo.compute_single_action(obs, policy_id="main")
                    # In case computer chooses an invalid action, pick a
                    # random one.
                    legal = time_step.observations["legal_actions"][player_id]
                    if action not in legal:
                        action = np.random.choice(legal)
                time_step = env.step([action])
                print(f"\n{env.get_state}")

            print(f"\n{env.get_state}")

            print("End of game!")
            if time_step.rewards[human_player] > 0:
                print("You win")
            elif time_step.rewards[human_player] < 0:
                print("You lose")
            else:
                print("Draw")
            # Switch order of players
            human_player = 1 - human_player

            num_episodes += 1

        algo.stop()

    ray.shutdown()