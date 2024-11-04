import gymnasium as gym
from ray.rllib import MultiAgentEnv
import soccer_twos, soccer_threes, soccer_fours
from ray.rllib.env import MultiAgentEnv
import numpy as np


class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass

class MultiAgentSoccerEnv(MultiAgentEnv):

    def __init__(self, env, num_per_team):
        super().__init__()
        # Do not start any threads or heavy processes in __init__
        self.env = env
        self.num_per_team = num_per_team
        # Define observation and action spaces (dummy spaces for now)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32
        )

        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])

    def reset(self, seed=0, options=None):
        return self.env.reset(), {}

    def step(self, action_dict):
        obs, reward, done, info = self.env.step(action_dict)
        truncs = {i: False for i in range(2*self.num_per_team)}
        truncs['__all__'] = False
        return obs, reward, done, truncs, {}

def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    num_per_team = env_config.get("num_per_team")
    if num_per_team == 4:
        unity_env = soccer_fours.make(**env_config)
    elif num_per_team == 3:
        unity_env = soccer_threes.make(**env_config)
    elif num_per_team == 2:
        unity_env = soccer_twos.make(**env_config)
    else:
        print(f"GOT UNEXPECTED NUMBER OF PLAYERS PER TEAM! {num_per_team}")
    multiagentenv = MultiAgentSoccerEnv(unity_env, num_per_team)
    return RLLibWrapper(multiagentenv)