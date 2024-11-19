import os
import logging
import argparse
from datetime import datetime

from torch.utils.data import Dataset, DataLoader

import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from config.base.attrdict import AttrDict
from data.h5_dataset import DacEncodecClapDatasetH5
from train import DiscodiffLitModel, ExceptionCallback, DemoCallback
from train import t5_padding_collate_func, check_config, save_config, logger
from utils import set_device_accelerator

def update_config(args, config: AttrDict):
    config_update = {}
    config_update["name"] = args.name
    config_update["testset_dir"] = args.testset_dir
    if args.num_gpus is not None:
        config_update["num_gpus"] = args.num_gpus
    if args.test_batch_size is not None:
        config_update["test_batch_size"] = args.test_batch_size
    config.override(config_update)

def main(args):
    # constuct save_path
    if args.name is None:
        args.name = datetime.now().strftime('testing-%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join("results", args.name)
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"- Created result path {save_path} \n")
    except:
        pass

    # load lightning module and config
    lightning_module = DiscodiffLitModel.load_from_checkpoint(args.ckpt_path)
    config = lightning_module.config
    update_config(args, config)
    check_config(config)
    logger.info(f"- Running with config \n {config} \n")
    config_save_path = os.path.join(save_path, "config.json")
    save_config(config_save_path, config)
    
    lightning_module.update_training_config(config)
    logger.info(f"- Lightning module initialized with given checkpoint {args.ckpt_path} \n")

    # callbacks
    exc_callback = ExceptionCallback()
    demo_callback = DemoCallback(config, save_path)
    
    # create data loader
    test_dataset = DacEncodecClapDatasetH5(
        h5_dir=config.testset_dir,
        dac_frame_len=config.sample_size,
        random_load=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0, 
        pin_memory=True,
        collate_fn=t5_padding_collate_func
    )
    logger.info("- Dataset created")

    # setup wandb logger
    if args.wandb_key is not None:
        wandb.login(key = args.wandb_key)
    wandb_logger = WandbLogger(project='discodiff_testing')
    wandb_logger.watch(lightning_module)
    
    # define trainer class for testing
    num_gpus = args.num_gpus if args.num_gpus >= 0 else 0
    force_cpu = (num_gpus == 0)
    device, accelerator = set_device_accelerator(force_cpu=force_cpu)
    strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"
    diffusion_trainer = L.Trainer(
        devices=num_gpus,
        accelerator=accelerator,
        strategy=strategy,
        precision=16,
        callbacks=[demo_callback, exc_callback],
        logger=wandb_logger,
    )
    logger.info("==============================================")
    logger.info("Lightning trainer initialized. Testing start.")
    
    # start training
    diffusion_trainer.test(lightning_module, dataloaders=test_dataloader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in testing Discodiff.')
    parser.add_argument(
        '-ckpt-path', type=str,
        help='the ckpt file; should be .ckpt'
    )
    parser.add_argument(
        '-testset-dir', type=str,
        help='the audio h5 dataset path for testing'
    )
    parser.add_argument(
        '--name', type=str, nargs='?',
        help='a string indicating the name of this run; if not specified, will be set to the timestamp'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help='the number of gpus; will use cpu if set to 0 or negative numbers; normally use 1 gpu'
    )
    parser.add_argument(
        '--test-batch-size', type=int, default=1,
        help='testing batch size under use; normally set to 1'
    )
    parser.add_argument(
        '--wandb-key', type=str, nargs='?',
        help='for login to wandb'
    )
    args = parser.parse_args()
    main(args)
