import gymnasium as gym
from ray.rllib import MultiAgentEnv
import soccer_twos, soccer_threes, soccer_fours
from ray.rllib.env import MultiAgentEnv
from gymnasium.spaces import Discrete, MultiDiscrete, Dict
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
        obs = self.env.reset()
        return obs, {}

    def step(self, action_dict):
        obs, reward, done, info = self.env.step(action_dict)
        truncs = {i: False for i in range(2*self.num_per_team)}
        truncs['__all__'] = False
        return obs, reward, done, truncs, {}
    
class MultiDiscreteToDiscreteActionWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Wrapper to convert MultiDiscrete action space to Discrete.
        
        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.observation_space = Dict({
            i: gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(336,), dtype=np.float32
            ) for i in range(8)
        })

        # Ensure the action space is MultiDiscrete
        if not isinstance(env.action_space, MultiDiscrete):
            raise ValueError("The action space must be MultiDiscrete.",type(env.action_space))

        self.multi_dim = env.action_space.nvec  # MultiDiscrete dimensions
        self.flat_action_space = int(np.prod(self.multi_dim))  # Total discrete actions

        # Replace the MultiDiscrete action space with Discrete
        self.action_space = Discrete(self.flat_action_space)

        self.action_space = Dict({
           i : Discrete(self.flat_action_space) for i in range(8)
        })

   # def _multidiscrete_to_discrete(self, action):
    #    """
     #   Convert MultiDiscrete action to a single Discrete index.
      #  """
       # return np.ravel_multi_index(action, self.multi_dim)

    def _discrete_to_multidiscrete(self, action):
        """
        Convert a single Discrete index back to MultiDiscrete action.
        """
        #print("ISSUEEEEE:",self.multi_dim, action)
        multi_discrete = {}
        for key in action.keys():
            multi_discrete[key] = np.unravel_index(action[key], self.multi_dim)
        #return np.unravel_index(action, self.multi_dim)
        return multi_discrete

    def reset(self, **kwargs):
        """
        Reset the environment (no changes to the observation space).
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Perform a step in the environment with converted action.
        
        Args:
            action: The action in Discrete format.
        
        Returns:
            A tuple (obs, reward, done, info) where the action was converted to MultiDiscrete.
        """
        # Convert Discrete action back to MultiDiscrete
        multidiscrete_action = self._discrete_to_multidiscrete(action)

        # Step in the environment with the MultiDiscrete action
        obs, reward, done, _, info = self.env.step(multidiscrete_action)
        return obs, reward, done, _, info

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

def create_rllib_env_flatten(env_config: dict = {}):
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
    multiagentenv = MultiDiscreteToDiscreteActionWrapper(MultiAgentSoccerEnv(unity_env, num_per_team))
    return RLLibWrapper(multiagentenv)