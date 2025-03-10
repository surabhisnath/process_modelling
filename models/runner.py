import argparse

def run(config):
    data = pd.read_csv("../csvs/" + config["dataset"] + ".csv")
    num_participants = len(data["pid"].unique())
    unique_responses = sorted(data["responses"].unique())  # 358 unique animals
    embeddings = get_embeddings(config, unique_responses)

    models = {} # nll function: min nll, betas, 
    if config["hills"]:
        Hills()
    
    for model_func in models:
        models[model_func]["minNLL"], models[model_func]["weights"] = fit(model_func)
        if config["test"]:
            simulate()
            test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="process_modelling", description="Implements various models of semantic exploration")

    parser.add_argument("--dataset", type=str, default="hills", help="claire or hills")
    parser.add_argument("--representation", type=str, default="clip", help="representation to use for embedding responses: ours, beagle, clip")
    parser.add_argument("--fitting", type=str, default="hierarchical", help="how to fit betas: individual, group or hierarchical")

    parser.add_argument("--hills", action="store_true", default=True, help="implement hills models (default: True)")
    parser.add_argument("--nohills", action="store_false", dest="hills", help="don't implement hills models")

    parser.add_argument("--morales", action="store_true", default=True, help="implement morales model (default: True)")
    parser.add_argument("--nomorales", action="store_false", dest="morales", help="don't implement morales models")

    parser.add_argument("--heineman", action="store_true", default=True, help="implement heineman models (default: True)")
    parser.add_argument("--noheineman", action="store_false", dest="heineman", help="don't implement heineman models")

    parser.add_argument("--abott", action="store_true", default=True, help="implement abott model (default: True)")
    parser.add_argument("--noabott", action="store_false", dest="abott", help="don't implement abott model")

    parser.add_argument("--our1", action="store_true", default=True, help="implement our class 1 models (default: True)")
    parser.add_argument("--noour1", action="store_false", dest="our1", help="don't implement our class 1 models")

    parser.add_argument("--our2", action="store_true", default=True, help="implement our class 2 models (default: True)")
    parser.add_argument("--noour2", action="store_false", dest="our2", help="don't implement our class 2 models")

    parser.add_argument("--test", action="store_true", default=True, help="test all models (default: True)")
    parser.add_argument("--notest", action="store_false", dest="test", help="don't test models")

    args = parser.parse_args()
    config = vars(args)

    run(config)