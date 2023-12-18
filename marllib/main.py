import argparse

from marllib import marl

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-e", "--env_name", type=str, default="macad")
    argparser.add_argument("-m", "--map_name", type=str, default="Town01")
    argparser.add_argument("-a", "--algo", type=str, default="mappo")
    argparser.add_argument("-c", "--core", type=str, default="gru")
    args = argparser.parse_args()

    # prepare env
    env = marl.make_env(environment_name=args.env_name, map_name=args.map_name)

    # initialize algorithm with appointed hyper-parameters
    algo_cls = getattr(marl.algos, args.algo, None)
    if algo_cls is None:
        raise ValueError("Unsupported algorithm specified: ", args.algo)
    else:
        try:
            algo = algo_cls(hyperparam_source=args.env_name)
        except:
            print("No finetuned hyper-parameters found, using default one...")
            algo = algo_cls(hyperparam_source="common")

    if args.algo.startswith("ma"):
        policy = "group"
    else:
        policy = "individual"

    # build agent model based on env + algorithms + user preference
    model = marl.build_model(env, algo, {"core_arch": args.core})

    # start training
    algo.fit(env, model, stop={"timesteps_total": 90000000}, share_policy=policy)
