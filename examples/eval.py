import ray
import yaml
import os
import sys
from ray import tune
from ray.tune import register_env
from ray.tune.utils import merge_dicts
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models import ModelCatalog
from marl.common import _get_model_config, recursive_dict_update, _get_config, recursive_dict_update, check_algo_type
from marl.models.zoo.cc_rnn import CC_RNN
from marl.models.zoo.ddpg_rnn import DDPG_RNN
from marl.algos.core.CC.mappo import MAPPOTrainer
from marl.algos.utils.setup_utils import AlgVar
from envs.base_env import ENV_REGISTRY
from envs.base_env import RllibMacad
from copy import deepcopy


def preparation(config_dict):
    ###################
    ### environment ###
    ###################

    env_reg_ls = []
    check_current_used_env_flag = False
    for env_n in ENV_REGISTRY.keys():
        if isinstance(ENV_REGISTRY[env_n], str):  # error
            info = [env_n, "Error", ENV_REGISTRY[env_n], "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
        else:
            info = [env_n, "Ready", "Null", "envs/base_env/config/{}.yaml".format(env_n),
                    "envs/base_env/{}.py".format(env_n)]
            env_reg_ls.append(info)
            if env_n == config_dict["env"]:
                check_current_used_env_flag = True

    if not check_current_used_env_flag:
        raise ValueError(
            "environment \"{}\" not installed properly or not registered yet, please see the Error_Log below".format(
                config_dict["env"]))

    map_name = config_dict["env_args"]["map_name"]
    test_env = ENV_REGISTRY[config_dict["env"]](config_dict["env_args"])
    agent_name_ls = test_env.agents
    env_info_dict = test_env.get_env_info()
    test_env.close()

    env_reg_name = config_dict["env"] + "_" + \
        config_dict["env_args"]["map_name"]
    register_env(env_reg_name,
                 lambda _: ENV_REGISTRY[config_dict["env"]](config_dict["env_args"]))

    #############
    ### model ###
    #############
    obs_dim = len(env_info_dict["space_obs"]["obs"].shape)

    if obs_dim == 1:
        encoder = "fc_encoder"
    else:
        encoder = "cnn_encoder"

    # load model config according to env_info:
    # encoder config
    encoder_arch_config = _get_model_config(encoder)
    config_dict = recursive_dict_update(config_dict, encoder_arch_config)

    # core rnn config
    rnn_arch_config = _get_model_config("rnn")
    config_dict = recursive_dict_update(config_dict, rnn_arch_config)

    ModelCatalog.register_custom_model(
        "Centralized_Critic_Model", CC_RNN)

    ModelCatalog.register_custom_model(
        "DDPG_Model", DDPG_RNN)

    ##############
    ### policy ###
    ##############

    policy_mapping_info = env_info_dict["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if config_dict["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError(
                "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

        policies = {"shared_policy"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "shared_policy")

    elif config_dict["share_policy"] == "group":
        groups = policy_mapping_info["team_prefix"]

        if len(groups) == 1:
            if not policy_mapping_info["all_agents_one_policy"]:
                raise ValueError(
                    "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

            policies = {"shared_policy"}
            policy_mapping_fn = (
                lambda agent_id, episode, **kwargs: "shared_policy")

        else:
            policies = {
                "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif config_dict["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(
            config_dict["share_policy"]))

    # if happo or hatrpo, force individual
    if config_dict["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info_dict["space_obs"], env_info_dict["space_act"], {}) for i in
            range(env_info_dict["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #####################
    ### common config ###
    #####################

    common_config = {
        # "seed": config_dict["seed"],
        "env": env_reg_name,
        "num_gpus_per_worker": config_dict["num_gpus_per_worker"],
        "num_gpus": config_dict["num_gpus"],
        "num_workers": config_dict["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": config_dict["framework"],
        "evaluation_interval": config_dict["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop = {
        "episode_reward_mean": config_dict["stop_reward"],
        "timesteps_total": config_dict["stop_timesteps"],
        "training_iteration": config_dict["stop_iters"],
    }

    return config_dict, common_config, env_info_dict, stop


def read_params():
    params = deepcopy(sys.argv)

    # convenient training
    webvis_flag = False
    for param in params:
        if param.startswith("--webvis"):
            webvis_flag = True
            ray_file_name = param.split("=")[1] + '.yaml'
            with open(os.path.join(os.path.dirname(__file__), "ray", ray_file_name), "r") as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                f.close()
    if not webvis_flag:
        with open(os.path.join(os.path.dirname(__file__), "ray/ray.yaml"), "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

    # env
    env_config = _get_config(params, "--env_config")
    config_dict = recursive_dict_update(config_dict, env_config)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
            config_dict["env_args"]["map_name"] = map_name

    # algorithm
    algo_type = ""
    for param in params:
        if param.startswith("--algo_config"):
            algo_name = param.split("=")[1]
            config_dict["algorithm"] = algo_name
            algo_type = check_algo_type(algo_name)

    algo_config = _get_config(params, "--algo_config", env_config)

    config_dict = recursive_dict_update(config_dict, algo_config)
    return config_dict


def get_trainer_config(config_dict, common_config, env_dict, stop):
    _param = AlgVar(config_dict)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = _param["batch_episode"] * env_dict["episode_limit"]
    if "fixed_batch_timesteps" in config_dict:
        train_batch_size = config_dict["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env_dict["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    # Fixed parameters
    batch_mode = _param["batch_mode"]
    use_gae = _param["use_gae"]
    num_sgd_iter = _param["num_sgd_iter"]
    lr_schedule = _param.get("lr_schedule", None)
    entropy_coeff_schedule = _param.get("entropy_coeff_schedule", None)

    lr = _param.get("lr", 1e-6)
    clip_param = _param.get("clip_param", 0.2)
    vf_clip_param = _param.get("vf_clip_param", 10.0)
    gae_lambda = _param.get("lambda", 0.99)
    kl_coeff = _param.get("kl_coeff", 0.2)
    vf_loss_coeff = _param.get("vf_loss_coeff", 1.0)
    entropy_coeff = _param.get("entropy_coeff", 0.01)

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr,
        "lr_schedule": lr_schedule,
        "entropy_coeff": entropy_coeff,
        "entropy_coeff_schedule": entropy_coeff_schedule,
        "num_sgd_iter": num_sgd_iter,
        "clip_param": clip_param,
        "use_gae": use_gae,
        "lambda": gae_lambda,
        "vf_loss_coeff": vf_loss_coeff,
        "kl_coeff": kl_coeff,
        "vf_clip_param": vf_clip_param,
        "model": {
            "custom_model": "Centralized_Critic_Model",
            "custom_model_config": merge_dicts(config_dict, env_dict),
        },
    }
    config.update(common_config)

    return config


if __name__ == '__main__':
    config_dict = read_params()

    ray.init(local_mode=config_dict["local_mode"])

    config_dict, common_config, env_info_dict, stop = preparation(config_dict)

    config = get_trainer_config(
        config_dict, common_config, env_info_dict, stop)

    agent = MAPPOTrainer(config=config)

    checkpoint = config_dict["algo_args"].get("restore", None)
    if checkpoint is not None:
        agent.restore(checkpoint)

    env = RllibMacad(config_dict["env_args"])
    obs = env.reset()
    states = {actor_id: agent.get_policy("shared_policy").get_initial_state() for actor_id in obs}

    done = {"__all__": False}
    while not done["__all__"]:
        action_dict = {}

        for actor in obs:
            action_dict[actor], states[actor], _ = agent.compute_single_action(obs[actor], states[actor], policy_id="shared_policy", explore=False)

        obs, reward, done, info = env.step(action_dict)

    env.close()
    ray.shutdown()

