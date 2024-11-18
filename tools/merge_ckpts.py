import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir("..")
import argparse
import torch
import lightning as L

from train import DiscodiffLitModel
from config.load_from_path import load_config_from_path
from utils import load_state_dict_partial_primary_secondary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in merging primary and secondary model.')
    parser.add_argument(
        '-config-path', type=str,
        help='the config file; should be .py or .json (for loading saved config)'
    )
    parser.add_argument(
        '-primary-ckpt-path', type=str,
    )
    parser.add_argument(
        '-secondary-ckpt-path', type=str,
    )
    parser.add_argument(
        '-out-ckpt-path', type=str
    )
    args = parser.parse_args()
    
    # get config object
    config = load_config_from_path(args.config_path)
    
    lightning_module = DiscodiffLitModel(config=config)
    load_state_dict_partial_primary_secondary(
        lightning_module.state_dict(),
        torch.load(args.primary_ckpt_path, map_location=torch.device('cpu'))['state_dict'],
        torch.load(args.secondary_ckpt_path, map_location=torch.device('cpu'))['state_dict'],
    )
    print("Loaded checkpoint", args.primary_ckpt_path, "and", args.secondary_ckpt_path)

    trainer = L.Trainer(accelerator="cpu")
    trainer.strategy.connect(lightning_module)
    trainer.save_checkpoint(args.out_ckpt_path)

    print("Saved checkpoint", args.out_ckpt_path)