from marllib import marl

# prepare env
env = marl.make_env(environment_name="macad", map_name="MeetCar")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='macad')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 9000000}, share_policy='group')

# This is a 1 iter fit, used as evaluation
# mappo.render(env, model, share_policy='group', restore_path={'model_path': '/home/morphlng/ray_results/mappo_gru_MeetCar_3090/MAPPOTrainer_macad_MeetCar_1d186_00000_0_2023-04-10_22-22-24/checkpoint_000200/checkpoint-200', 'params_path': '/home/morphlng/ray_results/mappo_gru_MeetCar_3090/MAPPOTrainer_macad_MeetCar_1d186_00000_0_2023-04-10_22-22-24/params.json'})