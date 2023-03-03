"""
    This file is a wrapper for Macad-Gym. 
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Tuple as GymTuple, Box
from macad_gym.envs import MultiCarlaEnv, HomoNcomIndePOIntrxMASS3CTWN3, HeteNcomIndePOIntrxMATLS1B2C1PTWN3
import numpy as np

env_name_mapping = {
    "HomoNcomIndePOIntrxMASS3CTWN3": HomoNcomIndePOIntrxMASS3CTWN3,
    "HeteNcomIndePOIntrxMATLS1B2C1PTWN3": HeteNcomIndePOIntrxMATLS1B2C1PTWN3,
    "default": MultiCarlaEnv,
    "custom": MultiCarlaEnv,
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "macad all scenarios",
        "team_prefix": ("car_",),
        # This means that all agents have the same policy
        "all_agents_one_policy": True,
        # This means that each agent has a different policy
        "one_agent_one_policy": True,
    },
}


class RllibMacad(MultiAgentEnv):
    def __init__(self, env_config):
        map_name = env_config.get("map_name", "default")
        env_class = env_name_mapping[map_name]
        if map_name != "custom":
            self.env = env_class()
            self.env_config = self.env.configs
        else:
            self.env = env_class(env_config)
            self.env_config = env_config

        self.num_agents = len(self.env_config["actors"].keys())
        self.agents = ["car_{}".format(i) for i in range(self.num_agents)]

        actor_id = next(iter(self.env_config["actors"].keys()))

        # TODO: macad does not have global state yet
        # Note1: obs is the partial observation of the agent, state is the global state
        # Note2: if enable send_measurements, the obs will be a Tuple of (obs, next_command, [speed, dist_to_goal])
        # We would use only obs as the observation
        obs_space = self.env.observation_space[actor_id]
        if isinstance(obs_space, GymTuple) or isinstance(obs_space, tuple):
            obs_space = obs_space[0]
            self.obs_with_measurement = True
        else:
            self.obs_with_measurement = False

        self.observation_space = GymDict({
            "obs": obs_space,
            # state represent other agent's obs
            "state": Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_space.shape[0], obs_space.shape[1], obs_space.shape[2] * (self.num_agents - 1)))
        })

        # Note: The action_space and observation_space here
        # represent single agent's action and observation space
        self.action_space = self.env.action_space[actor_id]

    def reset(self):
        origin_obs = self.env.reset()

        obs = {}
        for actor_id in origin_obs.keys():
            obs[actor_id] = {
                "obs": origin_obs[actor_id][0] if self.obs_with_measurement else origin_obs[actor_id],
                "state": np.concatenate([origin_obs[id][0] if self.obs_with_measurement else origin_obs[id]
                                         for id in origin_obs.keys()
                                         if id != actor_id], axis=-1)
            }

        return obs

    def step(self, action_dict):
        alive_actors = list(action_dict.keys())

        if len(alive_actors) < self.num_agents:
            for actor_id in self.env_config["actors"].keys():
                if actor_id not in alive_actors:
                    action_dict[actor_id] = 4  # brake

        origin_obs, r, d, i = self.env.step(action_dict)

        obs = {}
        for actor_id in origin_obs.keys():
            obs[actor_id] = {
                "obs": origin_obs[actor_id][0] if self.obs_with_measurement else origin_obs[actor_id],
                "state": np.concatenate([origin_obs[id][0] if self.obs_with_measurement else origin_obs[id]
                                        for id in origin_obs.keys()
                                        if id != actor_id], axis=-1)
            }

        # remove dead actor's observation
        for actor_id in list(obs.keys()):
            if actor_id not in alive_actors:
                del obs[actor_id]
                del r[actor_id]
                del d[actor_id]
                del i[actor_id]

        return obs, r, d, i

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env._scenario_config["max_steps"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
