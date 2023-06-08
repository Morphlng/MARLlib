import json

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.utils import merge_dicts

from marllib import marl
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.algos.utils.setup_utils import AlgVar
from marllib.marl.common import dict_update, recursive_dict_update


def get_file_config(algo, env, model, stop=None, **running_params):
    env_instance, info = env
    model_class, model_info = model

    algo.config_dict = info
    algo.config_dict = recursive_dict_update(algo.config_dict, model_info)

    algo.config_dict = recursive_dict_update(
        algo.config_dict, algo.algo_parameters)
    algo.config_dict = recursive_dict_update(algo.config_dict, running_params)

    algo.config_dict['algorithm'] = algo.name
    return algo.config_dict


def restore_config_update(exp_info, run_config, stop_config):
    if exp_info['restore_path']['model_path'] == '':
        restore_config = None
    else:
        restore_config = exp_info['restore_path']
        render_config = {
            "evaluation_interval": 1,
            "evaluation_num_episodes": 100,
            "evaluation_num_workers": 1,
            "evaluation_config": {
                "record_env": False,
                "render_env": True,
            }
        }

        run_config = recursive_dict_update(run_config, render_config)

        render_stop_config = {
            "training_iteration": 1,
        }

        stop_config = recursive_dict_update(stop_config, render_stop_config)

    return exp_info, run_config, stop_config, restore_config


def restore_model(restore, exp):

    if restore is not None:
        with open(restore["params_path"], 'r') as JSON:
            raw_exp = json.load(JSON)
            raw_exp = raw_exp["model"]["custom_model_config"]['model_arch_args']
            try:
                check_exp = exp["model"]["custom_model_config"]['model_arch_args']
            except:
                check_exp = exp["model_arch_args"]

            if check_exp != raw_exp:
                raise ValueError(
                    "is not using the params required by the checkpoint model")
        model_path = restore["model_path"]
    else:
        model_path = None

    return model_path


def get_exp_config(exp_info, env, model, stop=None):
    if not ray.is_initialized():
        ray.init(local_mode=exp_info["local_mode"])

    ########################
    ### environment info ###
    ########################

    env_info = env.get_env_info()
    map_name = exp_info['env_args']['map_name']
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()

    ######################
    ### policy sharing ###
    ######################

    policy_mapping_info = env_info["policy_mapping_info"]

    if "all_scenario" in policy_mapping_info:
        policy_mapping_info = policy_mapping_info["all_scenario"]
    else:
        policy_mapping_info = policy_mapping_info[map_name]

    if exp_info["share_policy"] == "all":
        if not policy_mapping_info["all_agents_one_policy"]:
            raise ValueError(
                "in {}, policy can not be shared, change it to 1. group 2. individual".format(map_name))

        policies = {"av"}
        policy_mapping_fn = (
            lambda agent_id, episode, **kwargs: "av")

    elif exp_info["share_policy"] == "group":
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
                "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
                groups
            }
            policy_ids = list(policies.keys())
            policy_mapping_fn = tune.function(
                lambda agent_id: "policy_{}_".format(agent_id.split("_")[0]))

    elif exp_info["share_policy"] == "individual":
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    else:
        raise ValueError("wrong share_policy {}".format(
            exp_info["share_policy"]))

    # if happo or hatrpo, force individual
    if exp_info["algorithm"] in ["happo", "hatrpo"]:
        if not policy_mapping_info["one_agent_one_policy"]:
            raise ValueError(
                "in {}, agent number too large, we disable no sharing function".format(map_name))

        policies = {
            "policy_{}".format(i): (None, env_info["space_obs"], env_info["space_act"], {}) for i in
            range(env_info["num_agents"])
        }
        policy_ids = list(policies.keys())
        policy_mapping_fn = tune.function(
            lambda agent_id: policy_ids[agent_name_ls.index(agent_id)])

    #########################
    ### experiment config ###
    #########################

    run_config = {
        "seed": int(exp_info["seed"]),
        "env": exp_info["env"] + "_" + exp_info["env_args"]["map_name"],
        "num_gpus_per_worker": exp_info["num_gpus_per_worker"],
        "num_gpus": exp_info["num_gpus"],
        "num_workers": exp_info["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn
        },
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }

    stop_config = dict_update(stop_config, stop)

    exp_info, run_config, stop_config, restore_config = restore_config_update(
        exp_info, run_config, stop_config)

    return exp_info, run_config, stop_config, restore_config


def get_trainer_config(model, exp, run, env, stop=None, restore=None):
    """ This script runs the Multi-Agent Proximal Policy Optimisation (MAPPO) algorithm using Ray RLlib.
    Args:
        :params model (str): The name of the model class to register.
        :params exp (dict): A dictionary containing all the learning settings.
        :params run (dict): A dictionary containing all the environment-related settings.
        :params env (dict): A dictionary specifying the condition for stopping the training.
        :params restore (bool): A flag indicating whether to restore training/rendering or not.

    Returns:
        ExperimentAnalysis: Object for experiment analysis.

    Raises:
        TuneError: Any trials failed and `raise_on_failed_trial` is True.
    """
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    ModelCatalog.register_custom_model(
        "Centralized_Critic_Model", model)

    _param = AlgVar(exp)

    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
    train_batch_size = _param["batch_episode"] * env["episode_limit"]
    if "fixed_batch_timesteps" in exp:
        train_batch_size = exp["fixed_batch_timesteps"]
    sgd_minibatch_size = train_batch_size
    episode_limit = env["episode_limit"]
    while sgd_minibatch_size < episode_limit:
        sgd_minibatch_size *= 2

    # Fixed parameters
    batch_mode = _param["batch_mode"]
    use_gae = _param["use_gae"]
    num_sgd_iter = _param["num_sgd_iter"]
    lr_schedule = _param.get("lr_schedule", None)
    entropy_coeff_schedule = _param.get("entropy_coeff_schedule", None)

    lr = _param.get("lr", 1e-5)
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
            "custom_model_config": merge_dicts(exp, env),
        },
    }
    config.update(run)

    return config


if __name__ == "__main__":
    # prepare env
    env = marl.make_env(environment_name="macad", map_name="Town01")
    env_instance, env_info = env

    # initialize algorithm with appointed hyper-parameters
    mappo = marl.algos.mappo(hyperparam_source='macad')

    # build agent model based on env + algorithms + user preference
    model = marl.build_model(env, mappo, {"core_arch": "gru"})
    model_class, model_info = model

    config_dict = get_file_config(mappo, env, model,
                                  share_policy='group', restore_path={
                                      'model_path': '/home/morphlng/ray_results/Town01_ckpt/Town01/checkpoint_000130/checkpoint-130',
                                      'params_path': '/home/morphlng/ray_results/Town01_ckpt/Town01/params.json'})

    exp_info, run_config, stop_config, restore_config = get_exp_config(
        config_dict, env_instance, model_class, stop=None)

    trainer_config = get_trainer_config(model_class, exp_info,
                                        run_config, env_instance.get_env_info(),
                                        stop_config, restore_config)

    agent = MAPPOTrainer(config=trainer_config)

    checkpoint = restore_model(restore_config, exp_info)
    if checkpoint is not None:
        agent.restore(checkpoint)


    # Inference
    obs = env_instance.reset()
    done = {"__all__": False}
    while not done["__all__"]:
        action_dict = {}
        for agent_id in obs.keys():
            action = agent.compute_single_action(obs[agent_id], policy_id="shared_policy")
            action_dict[agent_id] = action
        
        obs, reward, done, info = env_instance.step(action_dict)
    
    env.close()
    ray.shutdown()
    print("Inference finished!")
	