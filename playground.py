import torch
from train import DiscodiffLitModel
from config.load_from_path import load_config_from_path

config = load_config_from_path("config/configs_tiny.py")
lightning_module = DiscodiffLitModel(config=config)
print(lightning_module.state_dict().keys())
