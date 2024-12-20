import os
import sys
import json
import logging
import argparse
import importlib.util
from datetime import datetime

import torch
import torchaudio
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers import DPMSolverMultistepScheduler

import dac
from dac.model.dac import DAC

from typing import Union, Dict

from config.base.attrdict import AttrDict
from config.load_from_path import load_config_from_path
from models.unet.unet_1d_condition import UNet1DConditionModel
from models.unet.unet_1d_condition_simple import UNet1DConditionModelSimple
from pipelines.pipeline_discodiff import dac_latents_normalize, dac_latents_denormalize, DAC_DIM_SINGLE, DiscodiffPipeline
from utils import prob_mask_like, get_velocity, set_device_accelerator, pad_last_dim, audio_spectrogram_image
from train import logger, save_demo, save_config, check_config
from train import ExceptionCallback, DemoCallback, DacCLAPDataModule

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

class SimpleSecondaryLitModel(L.LightningModule):
    def __init__(self, config: AttrDict):
        super().__init__()

        self.config = config

        if config.architecture == "default":
            NNClass = UNet1DConditionModel
        else:
            NNClass = UNet1DConditionModelSimple
        
        # create denoise model and scheduler
        self.model_secondary = UNet1DConditionModel(
            sample_size=config.sample_size,
            in_channels=DAC_DIM_SINGLE,
            out_channels=config.out_channels_secondary,
            down_block_types=config.down_block_types_secondary,
            mid_block_type=config.mid_block_type_secondary,
            up_block_types=config.up_block_types_secondary,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels_secondary,
            encoder_hid_dim=config.encoder_hid_dim,
            encoder_hid_dim_type = "text_proj",
            time_cond_proj_dim=config.time_embedding_dim,
        )
        logger.info("-- Denoise model initialized --")
    
        # get other components
        dac_model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(dac_model_path).eval()
        for param in self.dac_model.parameters():
            param.requires_grad = False
        logger.info("-- Codec initialized --")
        
        self.update_training_config(config)
        
        self.loss_fn = torch.nn.L1Loss()
        self.debug = False # debug
        self.save_hyperparameters()

    def update_training_config(self, config: AttrDict):
        self.config = config
    
    def training_step(self, batch: Dict, batch_idx):
        # for checkpoint callback
        self.log('training_step', self.global_step)
        
        # create ground-truth normalized latents
        dac_latents = batch["dac_latents"] # [B, Kd, L]
        bs = dac_latents.shape[0]
        device = dac_latents.device
        dtype = dac_latents.dtype
        dac_latents_gt_primary = dac_latents_normalize(dac_latents[...,:DAC_DIM_SINGLE, :], selection="primary")
        dac_latents_gt_secondary = dac_latents_normalize(dac_latents[...,DAC_DIM_SINGLE:, :], selection="secondary")
        if self.debug:
            print(f"dac_latents_secondary mean  = {dac_latents[...,DAC_DIM_SINGLE:, :].mean().cpu().item()}, var = {dac_latents[...,DAC_DIM_SINGLE:, :].var().cpu().item()}")
            print(f"dac_latents_secondary normalized mean  = {dac_latents_gt_secondary.mean().cpu().item()}, var = {dac_latents_gt_secondary.var().cpu().item()}")
        
        # create model input
        input = dac_latents_gt_primary
        timesteps = torch.zeros((bs,), device=device).long()
        # create training target
        target = dac_latents_gt_secondary

        if self.debug:
            print(f"target mean  = {target.mean().cpu().item()}, var = {target.var().cpu().item()}")
        
        # training pass
        dummy_cond = torch.zeros((bs, 1, self.config.encoder_hid_dim)).to(device)
        model_output = self.model_secondary(input, timesteps, encoder_hidden_states=dummy_cond, return_dict=False)[0] # [B, (K-1)d, L]

        if self.debug:
            print(f"model_output mean  = {model_output.mean().cpu().item()}, var = {model_output.var().cpu().item()}")
        
        # compute loss
        main_loss = self.loss_fn(model_output, target)
        aux_loss = torch.abs(model_output.var() / target.var() - 1)
        if self.debug:
            print(f"main_loss = {main_loss.cpu().item()}, aux_loss = {aux_loss.cpu().item()}")
        loss = main_loss + aux_loss
        self.log('loss', loss, prog_bar=True)
        self.log('main_loss', main_loss, prog_bar=False)
        self.log('aux_loss', aux_loss, prog_bar=False)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        dac_latents = batch["dac_latents"] # [B, Kd, L]
        bs = dac_latents.shape[0]
        device = dac_latents.device
        dac_latents_gt_primary = dac_latents_normalize(dac_latents[...,:DAC_DIM_SINGLE, :], selection="primary")
        timesteps = torch.zeros((bs,), device=device).long()
        dummy_cond = torch.zeros((bs, 1, self.config.encoder_hid_dim)).to(device)
        secondary_given_primary = self.model_secondary(dac_latents_gt_primary, timesteps, encoder_hidden_states=dummy_cond, return_dict=False)[0]
        latents = torch.cat([dac_latents_gt_primary, secondary_given_primary], dim=1) # [B, Kd, L]
        final_latents = dac_latents_denormalize(latents)
        if self.debug:
            print(f"Generated latents", latents.mean().cpu().item(), latents.std().cpu().item()) # debug
            print(f"Generated final latents", final_latents.mean().cpu().item(), final_latents.std().cpu().item()) # debug
        
        z_sampled = self.dac_model.quantizer.from_latents(final_latents)[0] # [B, D, L]
        sampled_audios_given_primary = self.dac_model.decode(z_sampled).squeeze(1) # [B, T]
        if torch.abs(sampled_audios_given_primary).max() > 1:
            sampled_audios_given_primary = sampled_audios_given_primary / torch.abs(sampled_audios_given_primary).max()
        print("Reconstructing the ground-truth DAC latents...")

        # debug
        if self.debug:
            print("Ground-truth primary latent:", batch["dac_latents"][:,:DAC_DIM_SINGLE].mean().cpu().item(), batch["dac_latents"][:,:DAC_DIM_SINGLE].std().cpu().item())
            print("Ground-truth secondary latents:", batch["dac_latents"][:,DAC_DIM_SINGLE:].mean().cpu().item(), batch["dac_latents"][:,:DAC_DIM_SINGLE].std().cpu().item())
        
        z = self.dac_model.quantizer.from_latents(batch["dac_latents"])[0] # [B, D, L]
        sampled_audios_reconstructed = self.dac_model.decode(z).squeeze(1) # [B, T] # debug
        if torch.abs(sampled_audios_reconstructed).max() > 1:
            sampled_audios_reconstructed = sampled_audios_reconstructed / torch.abs(sampled_audios_reconstructed).max()
        
        out_dict = {
            "sampled_audios_given_primary": sampled_audios_given_primary,
            "sampled_audios_reconstructed": sampled_audios_reconstructed
        }
        return out_dict

    @torch.no_grad()
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx=dataloader_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model_secondary.parameters(), 
            lr=self.config.learning_rate
        )
        num_gpus = 1 if not hasattr(self.config, "num_gpus") else self.config.num_gpus
        if torch.cuda.is_available():
            print(f"There are {num_gpus} gpus available for training in this experiment.")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=int(self.config.train_dataset_size * self.config.max_epochs / self.config.train_batch_size / num_gpus),
        )
        lr_scheduler = {
            'scheduler': scheduler, # CosineAnnealingLR(optimizer, T_max=10, eta_min=self.config.learning_rate / 1e2),
            'interval': 'step', 
            'frequency': 1,
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

def update_config(args, config: AttrDict):
    config_update = {}
    config_update["name"] = args.name
    config_update["trainset_dir"] = args.trainset_dir
    config_update["valset_dir"] = args.valset_dir
    config_update["architecture"] = args.architecture
    if args.additional_trainset_dir is not None:
        config_update["additional_trainset_dir"] = args.additional_trainset_dir
    if args.train_batch_size is not None:
        config_update["train_batch_size"] = args.train_batch_size
    if args.num_val_demos is not None:
        config_update["val_dataset_size"] = args.num_val_demos
    if args.val_batch_size is not None:
        config_update["val_batch_size"] = args.val_batch_size
    if args.val_interval_steps is not None:
        config_update["val_interval_steps"] = args.val_batch_size
    if args.ckpt_save_top_k is not None:
        config_update["save_top_k"] = args.ckpt_save_top_k
    if args.num_gpus is not None:
        config_update["num_gpus"] = args.num_gpus
    config.override(config_update)

def main(args):
    # get config object
    config = load_config_from_path(args.config_path)

    # constuct save_path
    if args.name is None:
        args.name = datetime.now().strftime('training-%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join("results", args.name)
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            logger.info(f"- Created result path {save_path} \n")
    except:
        pass
        
    # renew config according to args and save the config
    update_config(args, config)
    
    check_config(config)
    logger.info(f"- Running with config \n {config} \n")
    config_save_path = os.path.join(save_path, "config.json")
    save_config(config_save_path, config)
    
    # create data module
    data_module = DacCLAPDataModule(config)
    logger.info("- Dataset created")

    # define lightning module
    if args.load_ckpt_path is not None:
        lightning_module = SimpleSecondaryLitModel.load_from_checkpoint(
            args.load_ckpt_path, config=config,
        )
        logger.info(f"- Lightning module initialized with given checkpoint {args.load_ckpt_path} \n")
    else:
        lightning_module = SimpleSecondaryLitModel(config=config)
        logger.info("- Lightning module initialized without checkpoint loading \n")
    lightning_module.update_training_config(config)

    # define callbacks
    exc_callback = ExceptionCallback()
    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=config.checkpoint_every, 
        save_top_k=config.save_top_k, 
        mode='max', 
        monitor='training_step',
        dirpath=save_path,
        filename='{epoch:04d}-{step}',
    )
    demo_callback = DemoCallback(config, save_path)

    # setup wandb logger
    if args.wandb_key is not None:
        wandb.login(key = args.wandb_key)
    # wandb_logger = WandbLogger(project='discodiff_training', log_model='all')
    wandb_logger = WandbLogger(project='discodiff_training', log_model=True)
    wandb_logger.watch(lightning_module)
    
    # define training class
    num_gpus = args.num_gpus if args.num_gpus >= 0 else 0
    force_cpu = (num_gpus == 0)
    device, accelerator = set_device_accelerator(force_cpu=force_cpu)
    strategy = "ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto"
    val_interval = config.demo_every if args.do_validation else None
    diffusion_trainer = L.Trainer(
        devices=num_gpus,
        accelerator=accelerator,
        strategy=strategy,
        precision=16,
        accumulate_grad_batches=1, 
        gradient_clip_val=0.5, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=val_interval, 
        check_val_every_n_epoch=None,
        max_epochs=config.max_epochs,
        profiler="simple"
    )
    logger.info("==============================================")
    logger.info("Lightning trainer initialized. Training start.")
    
    # start training
    diffusion_trainer.fit(lightning_module, data_module)

    # profile the time used to load data
    logger.info("Training stopped.")
    logger.info(f"total training visits: {data_module.train_dataset.total_visits}")
    logger.info(f"total train data loading time (sec): {data_module.train_dataset.total_runtime}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args in training Discodiff.')
    parser.add_argument(
        '-config-path', type=str,
        help='the config file; should be .py or .json (for loading saved config)'
    )
    parser.add_argument(
        '-trainset-dir', type=str,
        help='the audio h5 dataset path for training'
    )
    parser.add_argument(
        '--additional-trainset-dir', type=str, nargs='?',
        help='another audio h5 dataset path for training'
    )
    parser.add_argument(
        '--do-validation', type=bool, default=True,
        help='whether or not to do validation during training; default to False'
    )
    parser.add_argument(
        '--valset-dir', type=str, nargs='?',
        help='the audio h5 dataset path for validation; if not specified, will use training set to validate'
    )
    parser.add_argument(
        '--name', type=str, nargs='?',
        help='a string indicating the name of this run; if not specified, will be set to the timestamp'
    )
    parser.add_argument(
        '--architecture', type=str, default="default",
        help='to test on a better architecture: the conditional U-net better, or huggingface existing unconditional U-net better, choose from "default" and "huggingface"'
    )
    parser.add_argument(
        '--num-gpus', type=int, default=1,
        help='the number of gpus; will use cpu if set to 0 or negative numbers'
    )
    parser.add_argument(
        '--train-batch-size', type=int, nargs='?',
        help='training batch size under use; if not specified, then use config'
    )
    parser.add_argument(
        '--val-interval-steps', type=int, nargs='?',
        help='num of training steps between two validations; if not specified, then use config demo_every'
    )
    parser.add_argument(
        '--num-val-demos', type=int, nargs='?',
        help='number of chunks to load in validation; if not specified, then use config'
    )
    parser.add_argument(
        '--val-batch-size', type=int, nargs='?',
        help='validation batch size under use; if not specified, then use config'
    )
    parser.add_argument(
        '--load-ckpt-path', type=str, nargs='?',
        help='the checkpoint path to load'
    )
    parser.add_argument(
        '--ckpt-save-top-k', type=int, nargs='?',
        help='the number of ckpts to remain in training; if not specified, then use config'
    )
    parser.add_argument(
        '--wandb-key', type=str, nargs='?',
        help='for login to wandb'
    )
    args = parser.parse_args()
    main(args)
