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
        self.use_only_semantic = env_config.get("use_only_semantic", False)
        self.use_only_camera = env_config.get("use_only_camera", False)
        assert not (
            self.use_only_semantic and self.use_only_camera), "use_only_semantic and use_only_camera can not be True at the same time"

        env_class = env_name_mapping[map_name]
        if map_name != "custom":
            self.env = env_class()
            self.env_config = self.env.configs.copy()
        else:
            self.env = env_class(env_config)
            self.env_config = env_config.copy()

        if self.use_only_semantic:
            assert self.env_config["env"]["send_measurements"], "use_only_semantic can only be True when send_measurement is True"

        self.num_agents = len(self.env_config["actors"])
        self.agents = ["car_{}".format(i) for i in range(self.num_agents)]

        # Note1: obs is the partial observation of the agent, state is the global state
        # Note2: if enable send_measurements, the obs will be a Tuple of (obs, semantic_info)
        actor_id = next(iter(self.env_config["actors"].keys()))
        obs_space = self.env.observation_space[actor_id]

        if isinstance(obs_space, (GymTuple, tuple)):
            image_space = obs_space[0]
            self.obs_with_measurement = True
            if self.use_only_camera:
                obs_space = image_space
            elif self.use_only_semantic:
                obs_space = obs_space[1]
        else:
            self.obs_with_measurement = False
            image_space = obs_space

        obs_dict = {
            "obs": obs_space,
        }

        if self.use_only_camera:
            obs_dict.update({       # state represent other agent's obs
                "state": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(image_space.shape[0], image_space.shape[1], image_space.shape[2] * (self.num_agents - 1)))
            })

        self.observation_space = GymDict(obs_dict)

        # Note: The action_space and observation_space here
        # represent single agent's action and observation space
        self.action_space = self.env.action_space[actor_id]

    def _hard_reset(self):
        """If normal reset raise an exception, try hard reset first"""
        self.env.close()
        self.env = MultiCarlaEnv(self.env_config)
        return self.env.reset()

    def reset(self):

        try:
            origin_obs = self.env.reset()
        except Exception as e:
            print("Reset failed, try hard reset")
            origin_obs = self._hard_reset()

        obs = {}
        for actor_id in origin_obs.keys():
            if self.use_only_semantic:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id][1],
                }
            elif self.use_only_camera:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id][0] if self.obs_with_measurement else origin_obs[actor_id],
                    "state": np.concatenate([origin_obs[id][0] if self.obs_with_measurement else origin_obs[id]
                                             for id in origin_obs.keys()
                                             if id != actor_id], axis=-1)
                }
            else:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id]
                }

        return obs

    def step(self, action_dict):

        origin_obs, r, d, i = self.env.step(action_dict)

        obs = {}
        for actor_id in origin_obs.keys():
            if self.use_only_semantic:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id][1],
                }
            elif self.use_only_camera:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id][0] if self.obs_with_measurement else origin_obs[actor_id],
                    "state": np.concatenate([origin_obs[id][0] if self.obs_with_measurement else origin_obs[id]
                                             for id in origin_obs.keys()
                                             if id != actor_id], axis=-1)
                }
            else:
                obs[actor_id] = {
                    "obs": origin_obs[actor_id]
                }

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
