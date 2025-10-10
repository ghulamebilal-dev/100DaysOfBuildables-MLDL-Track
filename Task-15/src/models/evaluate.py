# Small helper, not strictly required because train_models does evaluation.
def pretty_metrics(metrics_dict):
    for model, m in metrics_dict.items():
        print(model)
        for k,v in m.items():
            print(f"  {k}: {v}")
