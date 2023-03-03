from ray import tune
from ray.tune.utils import merge_dicts
from ray.tune import CLIReporter
from marl.algos.core.CC.mappo import MAPPOTrainer
from marl.algos.utils.log_dir_util import available_local_dir
from marl.algos.utils.setup_utils import AlgVar


def run_mappo(config_dict, common_config, env_dict, stop):
    """
    for bug mentioned https://github.com/ray-project/ray/pull/20743
    make sure sgd_minibatch_size > max_seq_len
    """
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

    # Tunable parameters
    tuning = False
    lr = _param.get("lr", None)
    if lr is None:
        lr = tune.loguniform(1e-4, 1e-2)
        tuning = True

    clip_param = _param.get("clip_param", None)
    if clip_param is None:
        clip_param = tune.uniform(0.1, 0.5)
        tuning = True

    vf_clip_param = _param.get("vf_clip_param", None)
    if vf_clip_param is None:
        vf_clip_param = tune.uniform(10.0, 20.0)
        tuning = True

    gae_lambda = _param.get("lambda", None)
    if gae_lambda is None:
        gae_lambda = tune.uniform(0.95, 1.0)
        tuning = True

    kl_coeff = _param.get("kl_coeff", None)
    if kl_coeff is None:
        kl_coeff = tune.uniform(0.2, 0.5)
        tuning = True

    vf_loss_coeff = _param.get("vf_loss_coeff", None)
    if vf_loss_coeff is None:
        vf_loss_coeff = tune.uniform(0.5, 1.0)
        tuning = True

    entropy_coeff = _param.get("entropy_coeff", None)
    if entropy_coeff is None:
        entropy_coeff = tune.uniform(1e-3, 5e-2)
        tuning = True

    config = {
        "batch_mode": batch_mode,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "lr": lr,
        "lr_scedule": lr_schedule,
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

    algorithm = config_dict["algorithm"]
    map_name = config_dict["env_args"]["map_name"]
    arch = config_dict["model_arch_args"]["core_arch"]
    RUNNING_NAME = '_'.join(
        [algorithm, arch, map_name, "tune" if tuning else "train"])

    results = tune.run(MAPPOTrainer,
                       name=RUNNING_NAME,
                       stop=stop,
                       config=config,
                       num_samples=_param.get("num_samples", 1),
                       verbose=1,
                       progress_reporter=CLIReporter(),
                       local_dir=available_local_dir,
                       checkpoint_freq=_param.get("checkpoint_freq", 100),
                       checkpoint_at_end=True,
                       resume=_param.get("resume", False)
                       )

    if tuning:
        best_trial = results.get_best_trial(
            "episode_reward_mean", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation reward: {}".format(
            best_trial.last_result["episode_reward_mean"]))

    return results
