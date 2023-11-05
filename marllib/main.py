from marllib import marl

# prepare env
env = marl.make_env(environment_name="macad", map_name="Transport.Dynamic")

# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.mappo(hyperparam_source='macad')

# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "gru"})

# start training
mappo.fit(env, model, stop={'timesteps_total': 90000000}, share_policy='group')

# This is a 1 iter fit, used as evaluation
# mappo.render(env, model, share_policy='group', restore_path={
#     'model_path': '/home/morphlng/ray_results/Town01_ckpt/Town01/checkpoint_000130/checkpoint-130',
#     'params_path': '/home/morphlng/ray_results/Town01_ckpt/Town01/params.json'})