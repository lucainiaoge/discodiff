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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast

from typing import Union, Dict

from config.base.attrdict import AttrDict
from data.h5_dataset import DacEncodecClapDatasetH5
from models.unet.unet_1d_condition import UNet1DConditionModel
from pipelines.pipeline_discodiff import dac_latents_normalize, DAC_DIM_SINGLE, DiscodiffPipeline
from utils import prob_mask_like, get_velocity, set_device_accelerator, pad_last_dim, audio_spectrogram_image

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))

def display_chunk_data(chunk_datadict):
    for key in chunk_datadict.keys():
        this_component = chunk_datadict[key]
        if hasattr(this_component, "shape"):
            if len(this_component.shape) == 0:
                to_show = this_component
            else:
                to_show = f"shape = {this_component.shape}, max = {this_component.max()}, min = {this_component.min()}"
        else:
            to_show = this_component
    
        print(f"- {key}: {to_show}")

def t5_padding_collate_func(batch, debug = False):
    if debug:
        for chunk_datadict in batch:
            display_chunk_data(chunk_datadict)
            print("------")
    
    max_len = 0
    for i, this_dict in enumerate(batch):
        this_len = this_dict['t5_input_ids'].shape[-1]
        assert this_len == this_dict['t5_attention_mask'].shape[-1]
        if this_len > max_len:
            max_len = this_len

    for i, this_dict in enumerate(batch):
        this_len = this_dict['t5_input_ids'].shape[-1]
        if this_len < max_len:
            batch[i]["t5_input_ids"] = pad_last_dim(this_dict['t5_input_ids'], max_len, pad_val = 0)
            batch[i]["t5_null_input_ids"] = pad_last_dim(this_dict['t5_null_input_ids'], max_len, pad_val = 0)
            batch[i]["t5_attention_mask"] = pad_last_dim(this_dict['t5_attention_mask'], max_len, pad_val=0)

    return default_collate(batch)


class DacCLAPDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = DacEncodecClapDatasetH5(
            h5_dir=config.trainset_dir,
            dac_frame_len=config.sample_size,
            dataset_size=config.train_dataset_size,
            random_load=True,
        )
        
    # prepare dataset
    def setup(self, stage):
        pass

    # create train loader
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            num_workers=0, 
            pin_memory=True,
            collate_fn=t5_padding_collate_func
        )

    def val_dataloader(self):
        valsest_dir = self.config.valset_dir if self.config.valset_dir is not None else self.config.trainset_dir
        self.val_dataset = DacEncodecClapDatasetH5(
            h5_dir=valsest_dir,
            dac_frame_len=self.config.sample_size,
            dataset_size=self.config.val_dataset_size,
            random_load=True, # False, # debug
        )
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=0, 
            pin_memory=True,
            collate_fn=t5_padding_collate_func
        )
        
    def test_dataloader(self):
        self.test_dataset = DacEncodecClapDatasetH5(
            h5_dir=self.config.testset_dir,
            dac_frame_len=self.config.sample_size,
            random_load=False,
        )
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0, 
            pin_memory=True,
            collate_fn=t5_padding_collate_func
        )

class DiscodiffLitModel(L.LightningModule):
    def __init__(self, config: AttrDict):
        super().__init__()

        self.config = config

        # create denoise model and scheduler
        self.model_primary = UNet1DConditionModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels_primary,
            out_channels=config.out_channels_primary,
            down_block_types=config.down_block_types,
            mid_block_type=config.mid_block_type,
            up_block_types=config.up_block_types,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            num_class_embeds=config.num_class_embeds + 1, # the last class for CFG, this place is reserved for key conditioning
            class_embeddings_concat=config.class_embeddings_concat,
            encoder_hid_dim=config.encoder_hid_dim,
            encoder_hid_dim_type = "text_proj",
            time_cond_proj_dim=config.time_embedding_dim,
        )
        
        self.model_secondary = UNet1DConditionModel(
            sample_size=config.sample_size,
            in_channels=config.in_channels_secondary,
            out_channels=config.out_channels_secondary,
            down_block_types=config.down_block_types,
            mid_block_type=config.mid_block_type,
            up_block_types=config.up_block_types,
            layers_per_block=config.layers_per_block,
            block_out_channels=config.block_out_channels,
            num_class_embeds=config.num_class_embeds + 1, # # the last class for CFG, this place is reserved for key conditioning
            class_embeddings_concat=config.class_embeddings_concat,
            encoder_hid_dim=config.encoder_hid_dim,
            encoder_hid_dim_type = "text_proj",
            time_cond_proj_dim=config.time_embedding_dim,
        )
        
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            prediction_type=config.prediction_type,
        )
        logger.info("-- Denoise model and scheduler initialized --")
    
        # get other components
        dac_model_path = dac.utils.download(model_type="44khz")
        self.dac_model = dac.DAC.load(dac_model_path).eval()
        for param in self.dac_model.parameters():
            param.requires_grad = False
        
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(config.t5_model_name)
        for param in self.t5_model.parameters():
            param.requires_grad = False
        logger.info("-- Codec and text encoder initialized --")
        
        self.train_primary_prob = config.train_primary_prob
        assert self.train_primary_prob >= 0 and self.train_primary_prob <= 1

        self.load_audio_clap_prob = config.load_audio_clap_prob
        assert self.load_audio_clap_prob >= 0 and self.load_audio_clap_prob <= 1
        
        self.loss_fn = torch.nn.L1Loss()

        self.debug = False

        self.save_hyperparameters()

    @torch.no_grad()
    def get_text_emb_from_input_dict(self, batch: Dict):
        if "randomized_text" not in batch:
            return None, None
            
        text = batch["randomized_text"]
        null_text = batch["null_text"]
        if self.t5_model.training:
            self.t5_model.eval()
            torch.cuda.empty_cache()

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.float16):
            text_emb_seq = self.t5_model.encoder(
                input_ids=batch["t5_input_ids"],
                attention_mask=batch["t5_attention_mask"],
                return_dict=True,
            ).last_hidden_state # [B, L', D]
            
            null_text_emb_seq = self.t5_model.encoder(
                input_ids=batch["t5_null_input_ids"],
                attention_mask=batch["t5_attention_mask"],
                return_dict=True,
            ).last_hidden_state # [B, L', D]
            
        return text_emb_seq, null_text_emb_seq # [B, L', D]
    
    def training_step(self, batch: Dict, batch_idx):
        # for checkpoint callback
        self.log('training_step', self.global_step)
        
        # create ground-truth normalized latents
        train_primary = torch.rand(1) < self.train_primary_prob
        dac_latents = batch["dac_latents"] # [B, Kd, L]
        bs = dac_latents.shape[0]
        device = dac_latents.device
        dtype = dac_latents.dtype
        dac_latents_gt_primary = dac_latents_normalize(dac_latents[...,:DAC_DIM_SINGLE, :], selection="primary")
        dac_latents_gt_secondary = dac_latents_normalize(dac_latents[...,DAC_DIM_SINGLE:, :], selection="secondary") if not train_primary else None
        if self.debug:
            print(f"dac_latents mean  = {dac_latents.mean().cpu().item()}, var = {dac_latents.var().cpu().item()}")
            print(f"dac_latents_primary mean  = {dac_latents[...,:DAC_DIM_SINGLE, :].mean().cpu().item()}, var = {dac_latents[...,:DAC_DIM_SINGLE, :].var().cpu().item()}")
            print(f"dac_latents_secondary mean  = {dac_latents[...,DAC_DIM_SINGLE:, :].mean().cpu().item()}, var = {dac_latents[...,DAC_DIM_SINGLE:, :].var().cpu().item()}")
            print(f"dac_latents_primary normalized mean  = {dac_latents_gt_primary.mean().cpu().item()}, var = {dac_latents_gt_primary.var().cpu().item()}")
            if dac_latents_gt_secondary is not None:
                print(f"dac_latents_secondary normalized mean  = {dac_latents_gt_secondary.mean().cpu().item()}, var = {dac_latents_gt_secondary.var().cpu().item()}")
        
        # Sample a random timestep for each sample in batch
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=device
        ).long()
        
        # create model input
        noise = torch.randn(dac_latents.shape, device=device, dtype=dtype)
        if train_primary:
            noisy_input = self.noise_scheduler.add_noise(dac_latents_gt_primary, noise[...,:DAC_DIM_SINGLE, :], timesteps)
        else:
            noisy_input = self.noise_scheduler.add_noise(dac_latents_gt_secondary, noise[...,DAC_DIM_SINGLE:, :], timesteps)
            noisy_input = torch.cat((dac_latents_gt_primary, noisy_input), dim=1)

        # create training target
        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise[...,:DAC_DIM_SINGLE, :] if train_primary else noise[...,DAC_DIM_SINGLE:, :]
        elif self.noise_scheduler.config.prediction_type == 'sample':
            target = dac_latents_gt_primary if train_primary else dac_latents_gt_secondary
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            if train_primary:
                target = get_velocity(self.noise_scheduler, sample=dac_latents_gt_primary, noise=noise[...,:DAC_DIM_SINGLE, :], timesteps=timesteps)
            else:
                target = get_velocity(self.noise_scheduler, sample=dac_latents_gt_secondary, noise=noise[...,DAC_DIM_SINGLE:, :], timesteps=timesteps)
        else:
            assert 0, "Invalid scheduler prediction type"
        
        # get clap embedding
        load_audio_clap = torch.rand(1) < self.load_audio_clap_prob
        if load_audio_clap:
            clap_emb = batch["audio_clap"] if "audio_clap" in batch else None
        else:
            clap_emb = batch["randomized_text_clap"] if "randomized_text_clap" in batch else None
        null_clap_emb = batch["null_clap"]

        # get t5 embedding
        text_t5_embs, null_text_t5_embs = self.get_text_emb_from_input_dict(batch)
        text_t5_embs, null_text_t5_embs = text_t5_embs.to(dtype), null_text_t5_embs.to(dtype)

        # get key label if applicable
        null_class_labels = (torch.zeros(bs) + self.config.num_class_embeds).long().to(device)
        has_key_labels = "madmom_key" in batch
        class_labels = batch["madmom_key"] if has_key_labels else null_class_labels
        if self.config.key_cond_drop_prob > 0 and has_key_labels:
            keep_mask_labels = prob_mask_like((bs,), 1 - self.config.cfg_drop_prob, device=device)  # (B,)
            class_labels = torch.where(keep_mask_labels, class_labels, null_class_labels)

        # CFG randomly replace conditions with null conditions
        if self.config.clap_cond_drop_prob > 0:
            keep_mask_clap = prob_mask_like((bs, 1), 1 - self.config.clap_cond_drop_prob, device=device)  # (B, 1)
            clap_emb = torch.where(keep_mask_clap, clap_emb, null_clap_emb)
        if self.config.t5_cond_drop_prob > 0:
            keep_mask_t5 = prob_mask_like((bs, 1, 1), 1 - self.config.t5_cond_drop_prob, device=device)  # (B, 1, 1)
            text_t5_embs = torch.where(keep_mask_t5, text_t5_embs, null_text_t5_embs)
        
        # training pass
        model_to_train = self.model_primary if train_primary else self.model_secondary
        model_output = model_to_train(
            noisy_input, timesteps,
            encoder_hidden_states=text_t5_embs,
            class_labels=class_labels,
            timestep_cond=clap_emb,
            return_dict=False,
        )[0] # [B, d, L] (primary) or [B, (K-1)d, L] (secondary)
        
        # compute loss
        loss = self.loss_fn(model_output, target)

        # learning rate scheduler update
        self.lr_schedulers().step()
        self.log('loss', loss, prog_bar=True)
        
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        device = batch["dac_latents"].device

        # construct sampling pipeline
        if not hasattr(self, "pipeline"):
            self.pipeline = DiscodiffPipeline(
                dac_model=self.dac_model, 
                t5_model=self.t5_model, 
                tokenizer=self.tokenizer, 
                model_primary=self.model_primary, 
                model_secondary=self.model_secondary,
                scheduler=self.noise_scheduler,
            )
            self.pipeline.to_device(device)

        # get t5 embeddings
        prompt_embeds_t5, negative_prompt_embeds_t5 = self.get_text_emb_from_input_dict(batch)
        
        # get clap embedding
        prompt_embeds_clap_text = batch["randomized_text_clap"] if "randomized_text_clap" in batch else None
        negative_prompt_embeds_clap_text = batch["null_clap"]
        prompt_embeds_clap_audio = batch["audio_clap"] if "audio_clap" in batch else None

        # sample validation audio
        sampled_audios_default = self.pipeline(
            num_inference_steps = self.config.num_inference_timesteps,
            guidance_scale = self.config.cfg_scale,
            guidance_rescale = self.config.cfg_rescale,
            generator = torch.Generator(device=device).manual_seed(0),
            prompt_embeds_t5 = prompt_embeds_t5,
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5,
            prompt_embeds_clap_text = prompt_embeds_clap_text,
            negative_prompt_embeds_clap_text = negative_prompt_embeds_clap_text,
            prompt_embeds_clap_audio = prompt_embeds_clap_audio,
            use_audio_clap = False,
            return_dict = True,
        ).audios
        
        sampled_audios_given_audio_clap = self.pipeline(
            num_inference_steps = self.config.num_inference_timesteps,
            guidance_scale = self.config.cfg_scale,
            guidance_rescale = self.config.cfg_rescale,
            generator = torch.Generator(device=device).manual_seed(0),
            prompt_embeds_t5 = prompt_embeds_t5,
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5,
            prompt_embeds_clap_text = prompt_embeds_clap_text,
            negative_prompt_embeds_clap_text = negative_prompt_embeds_clap_text,
            prompt_embeds_clap_audio = prompt_embeds_clap_audio,
            use_audio_clap = True,
            return_dict = True,
        ).audios
        
        sampled_audios_given_primary = self.pipeline(
            num_inference_steps = self.config.num_inference_timesteps,
            guidance_scale = self.config.cfg_scale,
            guidance_rescale = self.config.cfg_rescale,
            generator = torch.Generator(device=device).manual_seed(0),
            
            primary_latents = batch["dac_latents"][...,:DAC_DIM_SINGLE, :], # default: unnormalized
            normalize_input_latents = True, # do normalization inside pipeline
            do_primary_loop = False,
            
            prompt_embeds_t5 = prompt_embeds_t5,
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5,
            prompt_embeds_clap_text = prompt_embeds_clap_text,
            negative_prompt_embeds_clap_text = negative_prompt_embeds_clap_text,
            prompt_embeds_clap_audio = prompt_embeds_clap_audio,
            use_audio_clap = False,
            return_dict = True,
        ).audios

        z = self.dac_model.quantizer.from_latents(batch["dac_latents"])[0] # [B, D, L]
        sampled_audios_reconstructed = self.dac_model.decode(z).squeeze(1) # [B, T] # debug
        if torch.abs(sampled_audios_reconstructed).max() > 1:
            sampled_audios_reconstructed = sampled_audios_reconstructed / torch.abs(sampled_audios_reconstructed).max()

        # if torch.rand(1) < 0.5:
        #     import matplotlib.pyplot as plt
        #     fig, ax = plt.subplots(1)
        #     ax.plot(sampled_audios_reconstructed[0].cpu().numpy()[:10000])
        #     fig.savefig("debug.png")
        #     plt.close(fig)
        
        out_dict = {
            "sampled_audios_default": sampled_audios_default,
            "sampled_audios_given_audio_clap": sampled_audios_given_audio_clap,
            "sampled_audios_given_primary": sampled_audios_given_primary,
            "sampled_audios_reconstructed": sampled_audios_reconstructed
        }
        return out_dict
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.model_primary.parameters()) + list(self.model_secondary.parameters()), 
            lr=self.config.learning_rate
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=int(self.config.train_dataset_size * self.config.max_epochs / self.config.train_batch_size),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }

def save_demo(
    outputs: Dict, 
    batch: Dict, 
    save_path: Union[str, os.PathLike], 
    sample_rate: int,
):
    bs = len(batch['name'])
    log_dict = {}
    tabel_columns = ["batch_id"]
    table_contents = []
    for i_batch in range(bs):
        audio_name = str(batch['name'][i_batch])
        filename_sample = f'{audio_name}.wav'
        this_table_content = [i_batch]

        if i_batch == 0:
            tabel_columns.append("name")
        this_table_content.append(audio_name)

        if 'text' not in batch:
            continue
        text_caption = str(batch['text'][i_batch])
        filename_text = f'{audio_name}.txt'
        dir_text = os.path.join(save_path, "text")
        if not os.path.exists(dir_text):
            os.makedirs(dir_text)
        filepath_text = os.path.join(dir_text, filename_text)
        with open(filepath_text, 'w') as f:
            f.write(text_caption)

        if i_batch == 0:
            tabel_columns.append("text")
        this_table_content.append(text_caption)
        log_dict[f'text_{i_batch}'] = text_caption
            
        for audio_type in outputs.keys():
            dir_sample = os.path.join(save_path, audio_type)
            if not os.path.exists(dir_sample):
                os.makedirs(dir_sample)
            filepath_sample = os.path.join(dir_sample, filename_sample)

            this_audio = outputs[audio_type][i_batch]
            if len(this_audio.shape) < 2:
                this_audio = this_audio.unsqueeze(0)
            this_wave_sample = this_audio.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            torchaudio.save(filepath_sample, this_wave_sample, sample_rate)
            
            if i_batch == 0:
                tabel_columns.append(f"wav_{audio_type}")
                tabel_columns.append(f"melspec_{audio_type}")
            this_table_content.append(
                wandb.Audio(
                    filepath_sample,
                    sample_rate=sample_rate,
                    caption=f'Audio {audio_name}, {audio_type}'
                )
            )
            log_dict[f"wav_{audio_type}_{i_batch}"] = this_table_content[-1]
            this_table_content.append(
                wandb.Image(
                    audio_spectrogram_image(this_audio),
                    caption=f'Melspec {audio_name}, {audio_type}'
                )
            )
            log_dict[f'melspec_{audio_type}_{i_batch}'] = this_table_content[-1]

        table_contents.append(this_table_content[:])

    log_table = {
        "data": table_contents,
        "columns": tabel_columns,
    }
    # log_table = wandb.Table(data=table_contents, columns=tabel_columns)
    return log_dict, log_table

def update_config(args, config: AttrDict):
    config_update = {}
    config_update["trainset_dir"] = args.trainset_dir
    config_update["valset_dir"] = args.valset_dir
    if args.train_batch_size is not None:
        config_update["train_batch_size"] = args.train_batch_size
    if args.num_val_demos is not None:
        config_update["val_dataset_size"] = args.num_val_demos
    if args.val_batch_size is not None:
        config_update["val_batch_size"] = args.val_batch_size
    if args.val_interval_steps is not None:
        config_update["val_interval_steps"] = args.val_batch_size
    if args.name is not None:
        config_update["name"] = args.name
    else:
        config_update["name"] = datetime.now().strftime('training-%Y-%m-%d-%H-%M-%S')
    if args.train_primary_prob is not None:
        config_update["train_primary_prob"] = args.train_primary_prob
    if args.load_audio_clap_prob is not None:
        config_update["load_audio_clap_prob"] = args.load_audio_clap_prob
    if args.ckpt_save_top_k is not None:
        config_update["save_top_k"] = args.ckpt_save_top_k
    config.override(config_update)

def save_config(save_path: Union[str, os.PathLike], config: AttrDict):
    with open(save_path, 'w') as f:
        json.dump(config.__dict__, f)
    
def check_config(config: AttrDict):
    if config.val_dataset_size % config.val_batch_size != 0:
        raise ValueError("Validation num demos should be multiplies of validation batch_size.")
    for key, value in config.__dict__.items():
        if "prob" in key and (value < 0 or value > 1):
            raise ValueError(f"{key} should be a probability with value taken in [0,1], but given {value}")
            

class ExceptionCallback(L.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}", file=sys.stderr)

class DemoCallback(L.Callback):
    def __init__(self, config: AttrDict, save_path: Union[str, os.PathLike], save_table = True):
        super().__init__()
        self.sample_rate = config.sample_rate
        self.last_demo_step = -1
        self.demo_save_path = save_path
        self.save_table = save_table

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
        log_dict, log_table = save_demo(outputs, batch, self.demo_save_path, self.sample_rate)
        if not self.save_table:
            trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        else:
            log_table_wandb = wandb.Table(data=log_table["data"], columns=log_table["columns"])
            # trainer.logger.log_table(key="samples", columns=log_table["columns"], data=log_table["data"], step=trainer.global_step)
            trainer.logger.experiment.log({"samples": log_table_wandb}, step=trainer.global_step)
            
    @torch.no_grad()
    def on_test_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.on_validation_batch_end(trainer, module, outputs, batch, batch_idx)

def main(args):
    # determine config type according to input
    config_dir, config_filename = os.path.split(args.config_path)
    config_name, config_ext = os.path.splitext(config_filename)
    if config_ext.lower() == ".py":
        sys.path.append(config_dir)
        spec=importlib.util.spec_from_file_location("config", args.config_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        config = foo.config
    elif config_ext.lower() == ".json":
        with open(args.config_path) as data_file:
            config_dict = json.load(data_file)
        config = AttrDict(config_dict)
    else:
        raise ValueError("Input config_path should be a .py file or a .json file")

    # constuct save_path
    save_path = os.path.join("results", args.name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        logger.info(f"- Created result path {save_path} \n")
        
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
        lightning_module = DiscodiffLitModel.load_from_checkpoint(
            args.load_ckpt_path,
        )
        logger.info(f"- Lightning module initialized with given checkpoint {args.load_ckpt_path} \n")
    else:
        lightning_module = DiscodiffLitModel(config=config)
        logger.info("- Lightning module initialized without checkpoint loading \n")

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
    diffusion_trainer = L.Trainer(
        devices=num_gpus,
        accelerator=accelerator,
        strategy=strategy,
        precision=16,
        accumulate_grad_batches=1, 
        callbacks=[ckpt_callback, demo_callback, exc_callback],
        logger=wandb_logger,
        log_every_n_steps=1,
        val_check_interval=config.demo_every, 
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
        '-valset-dir', type=str, nargs='?',
        help='the audio h5 dataset path for validation; if not specified, will use training set to validate'
    )
    parser.add_argument(
        '--name', type=str, nargs='?',
        help='a string indicating the name of this run; if not specified, will be set to the timestamp'
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
        '--train-primary-prob', type=float, nargs='?',
        help='the probability to train primary model in each training step; if not specified, then use config; if set to 0, then only train secondary model'
    )
    parser.add_argument(
        '--load-audio-clap-prob', type=float, nargs='?',
        help='the probability to load audio clap instead of text clap for training; if not specified, then use config'
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
