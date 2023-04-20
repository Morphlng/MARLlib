"""
    This file is a wrapper for Macad-Gym. 
"""
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Tuple as GymTuple, Box
from macad_gym.envs import MultiCarlaEnv, HomoNcomIndePOIntrxMASS3CTWN3, HeteNcomIndePOIntrxMATLS1B2C1PTWN3, Strike, MeetCarTown03, TestTown05
from copy import deepcopy
import numpy as np

env_name_mapping = {
    "HomoNcomIndePOIntrxMASS3CTWN3": HomoNcomIndePOIntrxMASS3CTWN3,
    "HeteNcomIndePOIntrxMATLS1B2C1PTWN3": HeteNcomIndePOIntrxMATLS1B2C1PTWN3,
    "Strike": Strike,
    "MeetCar": MeetCarTown03,
    "Town05": TestTown05,
    "default": MultiCarlaEnv,
    "custom": MultiCarlaEnv,
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "macad all scenarios",
        "team_prefix": ("car",),
        # This means that all agents have the same policy
        "all_agents_one_policy": True,
        # This means that each agent has a different policy
        "one_agent_one_policy": True,
    },
}


class RllibMacad(MultiAgentEnv):
    def __init__(self, env_config):
        config = deepcopy(env_config)
        self.map_name = config.get("map_name", "default")
        self.use_only_semantic = config.get("use_only_semantic", False)
        self.use_only_camera = config.get("use_only_camera", False)
        assert not (
            self.use_only_semantic and self.use_only_camera), "use_only_semantic and use_only_camera can not be True at the same time"

        env_class = env_name_mapping[self.map_name]
        if self.map_name != "custom":
            self.env = env_class()
        else:
            self.env = env_class(config)
        self.env_config = self.env.configs

        if self.use_only_semantic:
            assert self.env_config["env"]["send_measurements"], "use_only_semantic can only be True when send_measurement is True"

        self.agents = [actor_id for actor_id in self.env_config["actors"] if actor_id not in self.env._ignore_actor_ids and actor_id != "ego"]
        self.num_agents = len(self.agents)
        
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
        if self.env:
            self.env.close()
        
        env_class = env_name_mapping[self.map_name]
        if env_class == MultiCarlaEnv:
            self.env = MultiCarlaEnv(self.env_config)
        else:
            self.env = env_class()
        self.env_config = self.env.configs

    def reset(self):
        """Reset the environment and return the initial observation."""
        while True:
            try:
                origin_obs = self.env.reset()
                break
            except Exception as e:
                print("Exception raised when reset: {}".format(e))
                print("Reset failed, try hard reset")
                self._hard_reset()

        obs, _, _, _ = self._process_return(origin_obs)
        return obs

    def step(self, action_dict):

        # We add this only to update the observation
        # This action will not take effect
        if "ego" in self.env_config["actors"]:
            action_dict["ego"] = 0 if self.env_config["env"]["discrete_actions"] else (0, 0)

        try:
            origin_obs, r, d, i = self.env.step(action_dict)
        except Exception as e:
            print("Exception raised when step: {}".format(e))
            print("Step failed, set done to True and try hard reset on next reset")
            # Pseudo return
            obs, reward, done, info = self._process_return(self.env.observation_space.sample())
            self.env = None
            return obs, reward, done, info
            
        obs, reward, done, info = self._process_return(origin_obs, r, d, i)
        return obs, reward, done, info

    def _process_return(self, o, r=None, d=None, i=None):
        """Process the return of env.step"""
        obs, reward, done, info = {}, {}, {}, {}
        for actor_id in o.keys():
            if actor_id not in ["ego", "global"]:
                if self.use_only_semantic:
                    obs[actor_id] = {
                        "obs": o[actor_id][1],
                    }
                elif self.use_only_camera:
                    obs[actor_id] = {
                        "obs": o[actor_id][0] if self.obs_with_measurement else o[actor_id],
                        "state": np.concatenate([o[id][0] if self.obs_with_measurement else o[id]
                                                for id in o.keys()
                                                if id != actor_id], axis=-1)
                    }
                else:
                    obs[actor_id] = {
                        "obs": o[actor_id][1],
                        "state": o[actor_id][0]
                    }
                
                reward[actor_id] = r[actor_id] if r is not None else 0
                done[actor_id] = d[actor_id] if d is not None else True
                info[actor_id] = i[actor_id] if i is not None else None
        
        done["__all__"] = d["__all__"] if d is not None else True
        return obs, reward, done, info
        
    def close(self):
        self.env.close()

    def get_env_info(self):
        scenario_config = self.env._scenario_config

        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": scenario_config["max_steps"] if isinstance(scenario_config, dict) else scenario_config[0]["max_steps"],
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info
