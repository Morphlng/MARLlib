"""
Author: Morphlng
Date: 2023-08-09 19:34:29
LastEditTime: 2023-11-12 09:55:14
LastEditors: Morphlng
Description: Wrapper for macad env to restruct the observation and action space
FilePath: /MARLlib/marllib/envs/base_env/macad.py
"""

import logging
from copy import deepcopy

import numpy as np
from gym.spaces import Box, Dict
from macad_gym import ENV_ASSETS
from macad_gym.envs import *
from macad_gym.misc.experiment import ActionConcatWrapper, ActionPaddingWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv

logger = logging.getLogger(__name__)

env_name_mapping = {
    "Homo": HomoNcomIndePOIntrxMASS3CTWN3,
    "Hetero": HeteNcomIndePOIntrxMATLS1B2C1PTWN3,
    "Transport.Dynamic": transport.DynamicAvoidance,
    "Transport.Static": transport.StaticAvoidance,
    "Transport.Passthrough": transport.PassThrough,
    "Transport.Stopping": transport.Stopping,
    "Recce.Dynamic": recce.DynamicAvoidance,
    "Navigation": Navigation,
    "Town01": Town01Sim,
    "Town03": Town03Sim,
    "Town05": Town05Sim,
    "Town11": Town11Sim,
    "default": MultiCarlaEnv,
    "custom": MultiCarlaEnv,
}

policy_mapping_dict = {
    "all_scenario": {
        "description": "macad all scenarios",
        "team_prefix": ("agent_",),
        # This means that all agents have the same policy
        "all_agents_one_policy": True,
        # This means that each agent has a different policy
        "one_agent_one_policy": True,
    },
}


class RllibMacad(MultiAgentEnv):
    def __init__(self, env_config: dict):
        config = deepcopy(env_config)
        self.map_name = config.get("map_name", "default")
        self.use_only_semantic = config.get("use_only_semantic", False)
        self.use_only_camera = config.get("use_only_camera", False)
        self.pad_action_space = config.get("pad_action_space", False)
        self.concat_action_space = config.get("concat_action_space", False)

        if self.use_only_semantic and self.use_only_camera:
            raise ValueError(
                "`use_only_semantic` and `use_only_camera` can not be True at the same time"
            )

        env_class = env_name_mapping[self.map_name]
        self.env: MultiCarlaEnv = (
            env_class(config) if self.map_name == "custom" else env_class()
        )
        self.env_config = self.env.configs

        if self.use_only_semantic and not self.env.env_obs.send_measurements:
            raise ValueError(
                "`use_only_semantic` cannot be True when `send_measurement` is False"
            )

        self.agents = [
            actor_id
            for actor_id in self.env_config["actors"]
            if actor_id not in self.env._ignore_actor_ids and actor_id != "ego"
        ]
        self.num_agents = len(self.agents)

        if self.concat_action_space:
            logger.info("Enabling ActionConcatWrapper to homogenize the action space")
            self.env = ActionConcatWrapper(self.env)
        elif self.pad_action_space or self._check_heterogeneity():
            logger.info(
                "Heterogeneous agents detected, using ActionPaddingWrapper to pad the action space"
            )
            self.pad_action_space = True
            self.env = ActionPaddingWrapper(self.env)

        # Get observation space
        actor_id = next(iter(self.env_config["actors"].keys()))
        obs_space = self.env.observation_space[actor_id]

        if self.use_only_camera:
            image_space = obs_space["camera"]
            obs_dict = {
                "obs": image_space,
                "state": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        image_space.shape[0],
                        image_space.shape[1],
                        image_space.shape[2] * (self.num_agents - 1),
                    ),
                ),
            }
        elif self.use_only_semantic:
            obs_dict = {
                "obs": obs_space["state"],
            }
        else:
            obs_dict = {
                "obs": obs_space,
            }

        if "action_mask" in obs_space.spaces:
            self.use_mask_flag = True
            obs_dict.update({"action_mask": obs_space["action_mask"]})
        else:
            self.use_mask_flag = False

        self.observation_space = Dict(obs_dict)
        self.action_space = self.env.action_space[actor_id]
        self._last_obs = None

    def _hard_reset(self):
        """If normal reset raise an exception, try to hard reset the env"""
        if self.env:
            self.env.close()

        env_class = env_name_mapping[self.map_name]
        self.env = (
            MultiCarlaEnv(self.env_config)
            if env_class == MultiCarlaEnv
            else env_class()
        )
        self.env_config = self.env.configs

        if self.pad_action_space:
            self.env = ActionPaddingWrapper(self.env)
        elif self.concat_action_space:
            self.env = ActionConcatWrapper(self.env)

    def reset(self):
        """Reset the environment and return the initial observation."""
        for i in range(ENV_ASSETS.retries_on_error):
            try:
                origin_obs = self.env.reset()
                break
            except Exception:
                logger.exception("Exception raised during env.reset")
                logger.warning("Reset failed, try hard reset")
                self._hard_reset()
                if i == ENV_ASSETS.retries_on_error - 1:
                    raise RuntimeError("Maximum reset attempts exceeded")

        self._last_obs, _, _, _ = self._process_return(origin_obs)
        return self._last_obs

    def step(self, action_dict: dict):
        """Step the environment with the given action"""

        # We add this only to trigger the reward calculation for ego
        # Ego action will not take effect as long as using pseudo action (by default)
        if "ego" in self.env_config["actors"]:
            action_dict["ego"] = self.env.action_space["ego"].sample()

        try:
            origin_obs, r, d, i = self.env.step(action_dict)
        except Exception:
            logger.exception("Exception raised during env.step")
            logger.warning(
                "Step failed, set done to True and try hard reset on next reset"
            )
            # Pseudo return
            origin_obs, r, d, i = (
                self.env.observation_space.sample(),
                None,
                None,
                None,
            )
            self.env = None

        self._last_obs, reward, done, info = self._process_return(origin_obs, r, d, i)
        return self._last_obs, reward, done, info

    def _process_return(self, o, r=None, d=None, i=None):
        """Process the return of env.step"""
        obs, reward, done, info = {}, {}, {}, {}
        for actor_id in o.keys():
            if actor_id not in ["ego", "global"]:
                if self.use_only_semantic:
                    obs[actor_id] = {
                        "obs": o[actor_id]["state"],
                    }
                elif self.use_only_camera:
                    obs[actor_id] = {
                        "obs": o[actor_id]["camera"],
                        "state": np.concatenate(
                            [o[id]["camera"] for id in o.keys() if id != actor_id],
                            axis=-1,
                        ),
                    }

                if "action_mask" in o[actor_id]:
                    obs[actor_id]["action_mask"] = o[actor_id]["action_mask"]

                reward[actor_id] = r[actor_id] if r is not None else 0
                done[actor_id] = d[actor_id] if d is not None else True
                info[actor_id] = i[actor_id] if i is not None else None

        done["__all__"] = d["__all__"] if d is not None else True
        return obs, reward, done, info

    def _check_heterogeneity(self):
        """Check if the agents are heterogeneous"""
        spaces = {}
        for agent_action in self.env.agent_actions.values():
            action_type = agent_action.action_type
            if action_type not in spaces and "pseudo" not in action_type:
                spaces[action_type] = agent_action.space_type

        return len(spaces) > 1

    def close(self):
        self.env.close()

    def get_env_info(self):
        scenario_config = self.env.scenarios.scenario_config
        episode_limit = (
            scenario_config["max_steps"]
            if isinstance(scenario_config, dict)
            else scenario_config[0]["max_steps"]
        )

        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": episode_limit,
            "policy_mapping_info": policy_mapping_dict,
            "mask_flag": self.use_mask_flag,
        }
        return env_info
