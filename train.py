import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import MultiAgentEnv
import argparse
import soccer_fours, soccer_threes, soccer_twos
import gymnasium as gym
import numpy as np
from mlagents_envs.exception import UnityWorkerInUseException
import time

# Note: RLLib PPO Trainer doesn't support multidiscrete spaces so we have to convert between flattened discrete to multi discrete
# Use following functions to convert back and forth

# Mapping function to convert a Discrete action back to MultiDiscrete
def action_to_multidiscrete(action):
    return np.unravel_index(action, (3, 3, 3))

# Mapping function to convert a MultiDiscrete action to Discrete
def multidiscrete_to_action(action):
    return np.ravel_multi_index(action, (3, 3, 3))

class MultiAgentSoccerEnv(MultiAgentEnv):

    def __init__(self, config):
        super().__init__()
        # Do not start any threads or heavy processes in __init__
        self.unity_env = None  # Placeholder for the Unity environment
        self.num_per_team = config["num_per_team"]
        self.time_scale = config["time_scale"]
        self.render = False
        self.worker_id = 0
        self.max_retries = 5
        # Define observation and action spaces (dummy spaces for now)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(27)

    def reset(self, seed=0, options=None):
        if self.unity_env is None:
            # Initialize the Unity environment here, only when reset is called
            self._start_unity_environment()
        obs = self.unity_env.reset()
        info = {i: {} for i in range(2*self.num_per_team)}
        return obs, info

    def step(self, action_dict):
        if self.unity_env is None:
            print("Tried to step before initializing unity env!")
            return
        
        #convert flattened discrete to multi-discrete
        actions = {i: action_to_multidiscrete(action_dict[i]) for i in range(len(action_dict))}
        obs, reward, done, info = self.unity_env.step(actions)
        truncs = info = {i: False for i in range(2*self.num_per_team)}
        info = {i: {} for i in range(2*self.num_per_team)}
        truncs['__all__'] = False
        return obs, reward, done, truncs, info

    def _start_unity_environment(self):
        # Your code to start the Unity environment goes here
        for attempt in range(self.max_retries):
            try:
                if self.num_per_team == 4:
                    self.unity_env = soccer_fours.make(time_scale=self.time_scale, render=self.render, worker_id=self.worker_id)
                elif self.num_per_team == 3:
                    self.unity_env = soccer_threes.make(time_scale=self.time_scale, render=self.render, worker_id=self.worker_id)
                elif self.num_per_team == 2:
                    self.unity_env = soccer_twos.make(time_scale=self.time_scale, render=self.render, worker_id=self.worker_id)
                else:
                    print(f"GOT UNEXPECTED NUMBER OF PLAYERS PER TEAM! {num_per_team}")

                break  # Successfully started the environment
            except UnityWorkerInUseException:
                print(f"Worker ID {self.worker_id} is already in use. Retrying with a different worker ID...")
                self.worker_id += 1  # Increment the worker ID and try again
                time.sleep(1)  # Wait a bit before retrying
        else:
            raise RuntimeError("Failed to start Unity environment after multiple attempts.")
        
    def close(self):
        if self.unity_env is not None:
            self.unity_env.close()
            self.unity_env = None

def train_ppo(args):
    #Setup environment
    num_per_team = args.num_agents_per_team
    time_scale = args.timescale

    #hardcoded all envs observation and action space
    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32)
    action_space = gym.spaces.Discrete(27)


    #env = MultiAgentSoccerEnv(num_per_team, time_scale)

    # Initialize Ray
    ray.init()
    print("initialized ray")

    env_config = {
        "time_scale": time_scale,
        "num_per_team": num_per_team
    }

    multi_agent_config = {
            "policies": {
                "team_1_policy": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32),
                                gym.spaces.Discrete(27), {}),
                "team_2_policy": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32),
                                gym.spaces.Discrete(27), {}),
            },
            "policy_mapping_fn":lambda agent_id, *args, **kwargs: "team_1_policy" if agent_id in [0, 1] else "team_2_policy",
        }

    # Configure RLlib to use your custom environment
    config = (
        PPOConfig()
        .environment(env=MultiAgentSoccerEnv, env_config=env_config)
        .framework("torch")
        .multi_agent(**multi_agent_config)
    )

    # Build the PPO algorithm
    ppo = config.build()
    print("config built... starting train...")
    # Training loop
    for _ in range(100):
        result = ppo.train()
        print(result)

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Train multi-agent Soccer")

    # Add arguments
    #parser.add_argument('-', '--input', type=str, required=True, help='Input file path')
    #parser.add_argument('-o', '--output', type=str, help='Output file path', default='output.txt')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('-n', '--num_agents_per_team', type=int, choices=[2, 3, 4], 
        default=2, help='number of players per team in the soccer environmnet (must be either 2, 3, or 4)')
    parser.add_argument('--timescale', type=int, default=10, help='timescale for environment')

    # Parse the arguments
    args = parser.parse_args()

    if args.verbose:
        print(f"Using env Soccer{args.num_agents_per_team}v{args.num_agents_per_team}")
        print(f"Env timescale {args.timescale}")

    #Train agent
    train_ppo(args)

if __name__ == "__main__":
    main()