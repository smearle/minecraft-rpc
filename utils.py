
def get_exp_name(cfg):
    """Get the experiment name from the config"""
    exp_name = f"{cfg.model.name}_{cfg.data.name}_{cfg.data.num_samples}-data_{cfg.exp_name}"
    return exp_name