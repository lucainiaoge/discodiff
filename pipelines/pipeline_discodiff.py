import torch
import inspect
import numpy as np
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from typing import Optional, Tuple, Union, List, Callable

from dac.model.dac import DAC
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.t5.tokenization_t5_fast import T5TokenizerFast
from transformers import ClapModel, ClapProcessor
from models.unet.unet_1d_condition import UNet1DConditionModel
from utils import convert_audio

@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    audios: np.ndarray
    latents: Optional[torch.Tensor]

# the following codebook-specific normalization is deprecated
# but do not delete them, because the dimensions are of use
DAC_CODEBOOK_MEANS = [0.58, -0.11, -0.18, -0.04, -0.23, -0.06, 0.08, 0, 0.03] # [-0.0357, -0.0271, -0.0347, -0.0220, -0.0334, -0.0355, -0.0365, -0.0300, -0.0244]
DAC_CODEBOOK_STDS = [4.95, 4.00, 3.61, 3.41, 3.24, 3.16, 3.06, 2.93, 2.79] # [3.2464, 3.2818, 3.2376, 3.2643, 3.2814, 3.2808, 3.2847, 3.2798, 3.2684]
DAC_DIM_SINGLE = 8
DAC_N_CODEBOOKS = len(DAC_CODEBOOK_MEANS)
DAC_DIMS = DAC_DIM_SINGLE * DAC_N_CODEBOOKS
DAC_CODEBOOK_MEANS_BY_DIM = []
DAC_CODEBOOK_STDS_BY_DIM = []
for i_codebook in range(DAC_N_CODEBOOKS):
    DAC_CODEBOOK_MEANS_BY_DIM += [DAC_CODEBOOK_MEANS[i_codebook]]*DAC_DIM_SINGLE
    DAC_CODEBOOK_STDS_BY_DIM += [DAC_CODEBOOK_STDS[i_codebook]]*DAC_DIM_SINGLE
DAC_CODEBOOK_MEANS_BY_DIM = torch.tensor(DAC_CODEBOOK_MEANS_BY_DIM)
DAC_CODEBOOK_MEANS_PRIMARY_BY_DIM = DAC_CODEBOOK_MEANS_BY_DIM[:DAC_DIM_SINGLE]
DAC_CODEBOOK_MEANS_SECONDARY_BY_DIM = DAC_CODEBOOK_MEANS_BY_DIM[DAC_DIM_SINGLE:]
DAC_CODEBOOK_STDS_BY_DIM = torch.tensor(DAC_CODEBOOK_STDS_BY_DIM)
DAC_CODEBOOK_STDS_PRIMARY_BY_DIM = DAC_CODEBOOK_STDS_BY_DIM[:DAC_DIM_SINGLE]
DAC_CODEBOOK_STDS_SECONDARY_BY_DIM = DAC_CODEBOOK_STDS_BY_DIM[DAC_DIM_SINGLE:]

def shape_dac_mean_std(latents: torch.Tensor, selection: Optional[str] = None):
    shape = latents.shape
    if selection is None or selection == "all":
        mean = DAC_CODEBOOK_MEANS_BY_DIM[:]
        std = DAC_CODEBOOK_STDS_BY_DIM[:]
    elif selection == "primary":
        mean = DAC_CODEBOOK_MEANS_PRIMARY_BY_DIM[:]
        std = DAC_CODEBOOK_STDS_PRIMARY_BY_DIM[:]
    elif selection == "secondary":
        mean = DAC_CODEBOOK_MEANS_SECONDARY_BY_DIM[:]
        std = DAC_CODEBOOK_STDS_SECONDARY_BY_DIM[:]
    else:
        raise ValueError(
            f"Selection should be chosen from [None, 'all', 'primary', 'secondary'], but given {selection}."
        )
    
    if len(shape) == 3:
        dac_dims = shape[1]
        mean = mean.unsqueeze(0).unsqueeze(-1)
        std = std.unsqueeze(0).unsqueeze(-1)
    elif len(shape) == 2:
        dac_dims = shape[0]
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    elif len(shape) == 1:
        dac_dims = shape[0]
    else:
        raise ValueError(
            f"Invalid latents shape {shape}."
        )
    if selection is None or selection == "all":
        assert dac_dims == DAC_DIMS
    elif selection == "primary":
        assert dac_dims == DAC_DIM_SINGLE
    elif selection == "secondary":
        assert dac_dims == DAC_DIMS - DAC_DIM_SINGLE
        
    dtype = latents.dtype
    device = latents.device
    return mean.to(dtype = dtype, device = device), std.to(dtype = dtype, device = device)

def dac_latents_normalize_codebook_specific(latents: torch.Tensor, selection: Optional[str] = None):
    mean, std = shape_dac_mean_std(latents, selection=selection)
    print(mean.shape, std.shape)
    return (latents - mean) / std

def dac_latents_denormalize_codebook_specific(latents: torch.Tensor, selection: Optional[str] = None):
    mean, std = shape_dac_mean_std(latents, selection=selection)
    return latents * std + mean

DAC_NORMALIZE_FACTOR = 3.6
def dac_latents_normalize(latents: torch.Tensor, selection: Optional[str] = None):
    return latents / DAC_NORMALIZE_FACTOR

def dac_latents_denormalize(latents: torch.Tensor, selection: Optional[str] = None):
    return latents * DAC_NORMALIZE_FACTOR

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    # timesteps[0] = timesteps[0] - 1 # debug
    return timesteps, num_inference_steps

class DiscodiffPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        dac_model: DAC encoder decoder
        t5_model: Flan-T5 model for text encoding
        tokenizer: Tokenizer for t5 text encoder
        model_primary, model_secondary ([`UNet1DConditionModel`]): U-Net architecture to denoise the encoded audio in coarse-to-fine way.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded audio.
            Should be [`DPMSolverMultistepScheduler`]
    """

    def __init__(
        self, 
        dac_model: DAC, 
        t5_model: T5ForConditionalGeneration, 
        tokenizer: T5TokenizerFast, 
        model_primary: UNet1DConditionModel, 
        model_secondary: UNet1DConditionModel,
        scheduler: DPMSolverMultistepScheduler,
        scheduler_secondary: DPMSolverMultistepScheduler,
        clap_model: Optional[ClapModel] = None,
        clap_processor: Optional[ClapProcessor] = None,
    ):
        super().__init__()
        self.register_modules(
            dac_model=dac_model, 
            t5_model=t5_model, 
            tokenizer=tokenizer, 
            model_primary=model_primary, 
            model_secondary=model_secondary,
            scheduler=scheduler,
            scheduler_secondary=scheduler_secondary,
            clap_model=clap_model,
            clap_processor=clap_processor
        )

        if hasattr(model_secondary, "config"):
            assert self.model_primary.config.sample_size == self.model_secondary.config.sample_size
            if self.model_primary.config.time_cond_proj_dim is not None:
                assert self.model_primary.config.time_cond_proj_dim == self.model_secondary.config.time_cond_proj_dim

    @property
    def has_clap_model(self):
        return self._has_clap_model
    
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale
    
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
        
    @property
    def interrupt(self):
        return self._interrupt
    
    def encode_text_prompt(
        self,
        prompt: Optional[Union[str, List[str]]],
        num_audios_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds_t5: Optional[torch.Tensor] = None,
        negative_prompt_embeds_t5: Optional[torch.Tensor] = None,
        prompt_embeds_clap: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clap: Optional[torch.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_audios_per_prompt (`int`):
                number of audios that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the audio generation. If not defined, one has to pass
                `negative_prompt_embeds_t5` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds_t5 (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds_t5 (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds_t5 will be generated from `negative_prompt` input
                argument.
            similar for prompt_embeds_clap and negative_prompt_embeds_clap
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if prompt_embeds_t5 is not None:
            batch_size = prompt_embeds_t5.shape[0]
        elif prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt is None:
            batch_size = 1
            prompt = ""
        else:
            raise ValueError("Neither prompt nor prompt_embeds_t5 are provided. Should provide either one.")
            
        if prompt_embeds_clap is not None:
            assert prompt_embeds_clap.shape[0] == batch_size
        
        if self.has_clap_model:
            clap_device = next(self.clap_model.parameters()).device
        else:
            clap_device = next(self.model_primary.parameters()).device

        # get conditional t5 text embedding
        t5_device = next(self.t5_model.parameters()).device
        if prompt_embeds_t5 is None:
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because Flan-T5s can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )
            attention_mask = text_inputs.attention_mask.to(t5_device)
            
            prompt_embeds_t5 = self.t5_model.encoder(
                input_ids=text_input_ids.to(t5_device),
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        else:
            prompt_embeds_t5.to(t5_device)
        
        # get conditional clap text embedding
        if self.has_clap_model and prompt_embeds_clap is None:
            inputs_text_clap = self.clap_processor(text=prompt, return_tensors="pt").to(clap_device)
            prompt_embeds_clap = self.clap_model.get_text_features(**inputs_text_clap) # (n_texts, 512)
        elif prompt_embeds_clap is not None:
            pass
        else:
            prompt_embeds_clap = None

        # shape and dtype forming
        # if self.text_encoder is not None:
        #     prompt_embeds_dtype = self.text_encoder.dtype
        # else
        if self.model_primary is not None:
            prompt_embeds_dtype = self.model_primary.dtype
        else:
            prompt_embeds_dtype = prompt_embeds_t5.dtype

        prompt_embeds_t5 = prompt_embeds_t5.to(dtype=prompt_embeds_dtype, device=t5_device)

        bs_embed, seq_len, _ = prompt_embeds_t5.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds_t5 = prompt_embeds_t5.repeat(1, num_audios_per_prompt, 1)
        prompt_embeds_t5 = prompt_embeds_t5.view(bs_embed * num_audios_per_prompt, seq_len, -1)

        if prompt_embeds_clap is not None:
            prompt_embeds_clap = prompt_embeds_clap.repeat(1, num_audios_per_prompt)
            prompt_embeds_clap = prompt_embeds_clap.view(bs_embed * num_audios_per_prompt, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds_t5.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(t5_device)
            
            if negative_prompt_embeds_t5 is None:
                negative_prompt_embeds_t5 = self.t5_model.encoder(
                    input_ids=uncond_input.input_ids.to(t5_device),
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state
        
            if self.has_clap_model and negative_prompt_embeds_clap is None:
                uncond_inputs_text_clap = self.clap_processor(text=uncond_tokens, return_tensors="pt").to(clap_device)
                negative_prompt_embeds_clap = self.clap_model.get_text_features(**uncond_inputs_text_clap) # (n_texts, 512)
            elif negative_prompt_embeds_clap is not None:
                pass
            else:
                negative_prompt_embeds_clap = None

        # shape and dtype forming
        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds_t5.shape[1]

            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.to(dtype=prompt_embeds_dtype, device=t5_device)
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.repeat(1, num_audios_per_prompt, 1)
            negative_prompt_embeds_t5 = negative_prompt_embeds_t5.view(batch_size * num_audios_per_prompt, seq_len, -1)

            if negative_prompt_embeds_clap is not None:
                negative_prompt_embeds_clap = negative_prompt_embeds_clap.to(dtype=prompt_embeds_dtype, device=clap_device)
                negative_prompt_embeds_clap = negative_prompt_embeds_clap.repeat(1, num_audios_per_prompt)
                negative_prompt_embeds_clap = negative_prompt_embeds_clap.view(bs_embed * num_audios_per_prompt, -1)
        
        return prompt_embeds_t5, negative_prompt_embeds_t5, prompt_embeds_clap, negative_prompt_embeds_clap

    def encode_audio(
        self,
        wav: Optional[torch.Tensor],
        sample_rate: Optional[int] = None,
        num_audios_per_prompt: int = 1,
        audio_embeds_clap: Optional[torch.Tensor] = None,
    ):
        print("has_clap_model = ", self.has_clap_model)
        if (not self.has_clap_model or wav is None) and audio_embeds_clap is not None:
            return None, None

        if audio_embeds_clap is None:
            dtype = next(self.clap_model.parameters()).dtype
            device = next(self.clap_model.parameters()).device
            
            if wav is not None and sample_rate is None:
                raise ValueError("Sample rate must be provided to encode audio.")
            
            assert wav.dim() == 3, "The shape of input wav should be (batch, channels, n_samples)."
            
            sample_rate_clap = self.clap_processor.feature_extractor.sampling_rate
            wav_clap = convert_audio(wav, sample_rate, sample_rate_clap, 1) # (B, C, T) -> (B, 1, T')
            wav_clap = wav_clap.squeeze(1)
            clap_inputs = self.clap_processor(audios=wav_clap.cpu().numpy(), sampling_rate=sample_rate_clap, return_tensors="pt").to(device=device)
            audio_embeds_clap = self.clap_model.get_audio_features(**clap_inputs) # (B, 512)
        
        audio_embeds_clap = audio_embeds_clap.repeat_interleave(num_audios_per_prompt, dim=0)
        uncond_audio_embeds_clap = torch.zeros_like(audio_embeds_clap)

        return audio_embeds_clap, uncond_audio_embeds_clap

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        sample_size: int,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        
        prompt_embeds_t5: Optional[torch.Tensor] = None,
        negative_prompt_embeds_t5: Optional[torch.Tensor] = None,
        prompt_embeds_clap_text: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clap_text: Optional[torch.Tensor] = None,
        prompt_embeds_clap_audio: Optional[torch.Tensor] = None,

        prompt_wav: Optional[torch.Tensor] = None,
    ):
        if sample_size % 8 != 0:
            raise ValueError(f"`sample_size` (frame len) have to be divisible by 8 but is {sample_size}.")

        if prompt is not None and prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds_t5`: {prompt_embeds_t5}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds_t5 is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds_t5`. Cannot leave both `prompt` and `prompt_embeds_t5` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_wav is not None and prompt_embeds_clap_audio is not None:
            raise ValueError(
                f"Cannot forward both `prompt_wav`: {prompt_wav} and `prompt_embeds_clap_audio`: {prompt_embeds_clap_audio}. Please make sure to"
                " only forward one of the two."
            )
        
        if negative_prompt is not None and negative_prompt_embeds_t5 is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds_t5`:"
                f" {negative_prompt_embeds_t5}. Please make sure to only forward one of the two."
            )

        if prompt_embeds_t5 is not None and negative_prompt_embeds_t5 is not None:
            if prompt_embeds_t5.shape != negative_prompt_embeds_t5.shape:
                raise ValueError(
                    "`prompt_embeds_t5` and `negative_prompt_embeds_t5` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_t5` {prompt_embeds_t5.shape} != `negative_prompt_embeds_t5`"
                    f" {negative_prompt_embeds_t5.shape}."
                )
                
        if prompt_embeds_clap_text is not None and negative_prompt_embeds_clap_text is not None:
            if prompt_embeds_clap_text.shape != negative_prompt_embeds_clap_text.shape:
                raise ValueError(
                    "`prompt_embeds_clap_text` and `negative_prompt_embeds_clap_text` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_clap_text` {prompt_embeds_clap_text.shape} != `negative_prompt_embeds_clap_text`"
                    f" {negative_prompt_embeds_clap_text.shape}."
                )

        if prompt_embeds_clap_text is not None and prompt_embeds_clap_audio is not None:
            if prompt_embeds_clap_text.shape != prompt_embeds_clap_audio.shape:
                raise ValueError(
                    "`prompt_embeds_clap_text` and `prompt_embeds_clap_audio` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds_clap_text` {prompt_embeds_clap_text.shape} != `prompt_embeds_clap_audio`"
                    f" {prompt_embeds_clap_audio.shape}."
                )

    # can use neither or either clap text or audio embedding. Cannot use both.
    def check_clap_embs(
        self, 
        prompt_embeds_clap_text: Optional[torch.Tensor] = None,
        prompt_embeds_clap_audio: Optional[torch.Tensor] = None,
    ):
        use_clap_text = False
        use_clap_audio = False
        if prompt_embeds_clap_text is not None:
            use_clap_text = True
        if prompt_embeds_clap_audio is not None:
            use_clap_audio = True
        if use_clap_text and use_clap_audio:
            use_clap_text = False
        return use_clap_text, use_clap_audio
    
    def prepare_latents(
        self, 
        batch_size, 
        num_channels_latents, 
        sample_size, 
        dtype, 
        device, 
        generator, 
        latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            sample_size,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        else:
            latents = latents.to(device)
            
        return latents

    # def prepare_class_labels(
    #     self, 
    #     batch_size,
    #     device,    
    #     num_audios_per_prompt:int = 1, 
    #     class_labels: Optional[torch.LongTensor] = None,
    #     do_classifier_free_guidance: bool = True,
    # ):
    #     if class_labels is None:
    #         # TODO: rand sample or always null?
    #         class_labels = torch.randint(
    #             low=0, high=self.model_primary.config.num_class_embeds - 1, size=(batch_size,)
    #         ).long().to(self.device)
            
    #         class_labels = torch.ones(batch_size).long().to(self.device) * (self.model_primary.config.num_class_embeds - 1) # debug
    #         # return None
    #     else:
    #         class_labels = class_labels.long().to(self.device)
    #         class_labels = class_labels.repeat_interleave(num_audios_per_prompt, dim=0)
    #         null_class_labels = (class_labels - class_labels) + self.model_primary.config.num_class_embeds - 1  # already +1 in unet
    #         class_labels = torch.cat([null_class_labels, class_labels], dim=0) if do_classifier_free_guidance else class_labels
    #     return class_labels

    def prepare_class_labels(
        self, 
        batch_size,
        device,    
        num_audios_per_prompt:int = 1, 
        class_labels: Optional[torch.LongTensor] = None,
        do_classifier_free_guidance: bool = True,
    ):
        if class_labels is None:
            return None
        else:
            class_labels = class_labels.long().to(self.device)
            class_labels = class_labels.repeat_interleave(num_audios_per_prompt, dim=0)
            null_class_labels = (class_labels - class_labels) + self.model_primary.config.num_class_embeds - 1  # already +1 in unet
            class_labels = torch.cat([null_class_labels, class_labels], dim=0) if do_classifier_free_guidance else class_labels
            return class_labels
    
    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32, scale_factor: float = 0.1
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb * scale_factor

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def to_device(self, device):
        self.dac_model.to(device)
        self.t5_model.to(device)
        self.model_primary.to(device)
        self.model_secondary.to(device)
        if self.clap_model is not None:
            self.clap_model.to(device)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_audios_per_prompt: Optional[int] = 1,
        class_labels: Optional[torch.LongTensor] = None, # not supported currently
        
        sample_size: int = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        guidance_rescale: float = 0.0,
        eta: float = 0.0,  # for DDIM only
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,

        primary_latents: Optional[torch.Tensor] = None,
        secondary_latents: Optional[torch.Tensor] = None,
        normalize_input_latents: bool = True,
        do_primary_loop: bool = True,
        
        prompt_embeds_t5: Optional[torch.Tensor] = None,
        negative_prompt_embeds_t5: Optional[torch.Tensor] = None,
        prompt_embeds_clap_text: Optional[torch.Tensor] = None,
        negative_prompt_embeds_clap_text: Optional[torch.Tensor] = None,
        prompt_embeds_clap_audio: Optional[torch.Tensor] = None,

        prompt_wav: Optional[torch.Tensor] = None,
        prompt_wav_sample_rate: Optional[int] = None,

        use_audio_clap: bool = True,
        use_old_model = False,
        
        return_dict: bool = True,
        **kwargs,
    ) -> Union[AudioPipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, you need to pass `prompt_embeds`.
                Otherwise it will be unconditioned
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in audio generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
                Otherwise it will be set to [""]
            num_audios_per_prompt (`int`, *optional*, defaults to 1):
                The number of audios to generate per prompt.
                
            sample_size (`int`, *optional*, defaults to `self.model_primary.config.sample_size`):
                The num of dac frames of generated latents
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality audio at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate latents closely linked to the text
                `prompt` at the expense of lower quality. Guidance scale is enabled when `guidance_scale > 1`.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.

            primary_latents (`torch.Tensor`, *optional*):
                Pre-generated noisy primary_latents sampled from a Gaussian distribution, to be used as inputs for latent
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            secondary_latents (`torch.Tensor`, *optional*):
                Pre-generated noisy secondary_latents sampled from a Gaussian distribution, to be used as inputs for latent
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            normalize_input_latents (`bool`, defaults to True):
                If assuming the input latents are unnormalized, set it to True, and they will be normalized first before
                being shaped as latents input
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated latent. `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.AudioPipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.AudioPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.AudioPipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated audios
        """
        self.debug = False

        # 0. Default sample_size to unet
        sample_size = sample_size or self.model_primary.config.sample_size

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            sample_size=sample_size,
            negative_prompt=negative_prompt,
            prompt_embeds_t5=prompt_embeds_t5,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            prompt_embeds_clap_text=prompt_embeds_clap_text,
            negative_prompt_embeds_clap_text=negative_prompt_embeds_clap_text,
            prompt_embeds_clap_audio=prompt_embeds_clap_audio,
            prompt_wav=prompt_wav,
        )
        if not do_primary_loop and primary_latents is None:
            raise ValueError(
                f"Cannot run secondary-only sampling when primary latents are not provided."
            )
            
        self._has_clap_model = True if (self.clap_model is not None and self.clap_processor is not None) else False
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._interrupt = False

        timesteps_given = timesteps
        
        num_channels_latents_primary = self.model_primary.config.in_channels
        dim_codebook = num_channels_latents_primary
        if hasattr(self.model_secondary, "config"):
            num_in_channels_latents_secondary = self.model_secondary.config.in_channels
        else:
            num_in_channels_latents_secondary = 72
        num_channels_latents_secondary = num_in_channels_latents_secondary - num_channels_latents_primary
        n_codebooks = num_in_channels_latents_secondary // dim_codebook
        assert num_in_channels_latents_secondary % dim_codebook == 0
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds_t5.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        prompt_embeds_t5, negative_prompt_embeds_t5, prompt_embeds_clap, negative_prompt_embeds_clap = self.encode_text_prompt(
            prompt=prompt,
            num_audios_per_prompt=num_audios_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds_t5=prompt_embeds_t5,
            negative_prompt_embeds_t5=negative_prompt_embeds_t5,
            prompt_embeds_clap=prompt_embeds_clap_text,
            negative_prompt_embeds_clap=negative_prompt_embeds_clap_text,
        )
        audio_embeds_clap, uncond_audio_embeds_clap = self.encode_audio(
            wav=prompt_wav,
            sample_rate=prompt_wav_sample_rate,
            num_audios_per_prompt=num_audios_per_prompt,
            audio_embeds_clap=prompt_embeds_clap_audio,
        )
        has_text_clap_input, has_audio_clap_input = self.check_clap_embs(
            prompt_embeds_clap_text=prompt_embeds_clap,
            prompt_embeds_clap_audio=audio_embeds_clap,
        )
        if has_audio_clap_input and use_audio_clap:
            prompt_embeds_clap = audio_embeds_clap
            negative_prompt_embeds_clap = uncond_audio_embeds_clap

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds_t5 = torch.cat([negative_prompt_embeds_t5, prompt_embeds_t5])
            if prompt_embeds_clap is not None:
                prompt_embeds_clap = torch.cat([negative_prompt_embeds_clap, prompt_embeds_clap])
        
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps_given, sigmas
        )
        
        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 5. Prepare latent variables (primary)
        if primary_latents is not None:
            primary_latents = dac_latents_normalize(primary_latents, selection="primary") if normalize_input_latents else primary_latents
            if do_primary_loop:
                do_primary_loop = False
                print("As primary_latents is given, do_primary_loop is automatically set to False.")
        else:
            do_primary_loop = True

        primary_latents = self.prepare_latents(
            batch_size * num_audios_per_prompt,
            num_channels_latents_primary,
            sample_size,
            prompt_embeds_t5.dtype,
            device,
            generator,
            primary_latents,
        )

        if self.debug:
            print("primary_latents before loop", primary_latents.mean().cpu().item(), primary_latents.var().cpu().item()) # debug

        
        '''
        # 6. Optionally get Guidance Scale Embedding, not supported currently
        guidance_scale_emb = None
        if self.model_primary.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_audios_per_prompt)
            guidance_scale_emb = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model_primary.config.time_cond_proj_dim
            ).to(device=device, dtype=primary_latents.dtype).repeat(2, 1) # cfg has doubled input
        '''
        # form input class conditions (not supported currently)
        class_labels = self.prepare_class_labels(
            batch_size=batch_size, device=device, 
            num_audios_per_prompt=num_audios_per_prompt, 
            class_labels=class_labels,
            do_classifier_free_guidance=self.do_classifier_free_guidance
        )

        # 6. Denoising loop (primary)
        self._num_timesteps = len(timesteps)
        for t in self.progress_bar(timesteps):
            if not do_primary_loop:
                break
            
            # expand the latents if we are doing classifier free guidance
            primary_latent_input = torch.cat([primary_latents] * 2) if self.do_classifier_free_guidance else primary_latents
            primary_latent_input = self.scheduler.scale_model_input(primary_latent_input, t)
            
            # predict the noise residual
            primary_pred = self.model_primary(
                primary_latent_input, t,
                encoder_hidden_states=prompt_embeds_t5,
                class_labels=class_labels,
                timestep_cond=prompt_embeds_clap,
                return_dict=True,
            ).sample # [B, d, L]

            # perform guidance
            if self.do_classifier_free_guidance:
                primary_pred_uncond, primary_pred_cond = primary_pred.chunk(2)
                primary_pred = primary_pred_uncond + self.guidance_scale * (primary_pred_cond - primary_pred_uncond)

            if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                primary_pred = rescale_noise_cfg(primary_pred, primary_pred_cond, guidance_rescale=self.guidance_rescale)

            if t == timesteps[1] or t == timesteps[-1] and self.debug:
                print(f"Primary loop: primary_pred at timestep {t}", primary_pred.mean().cpu().item(), primary_pred.var().cpu().item()) # debug

            # compute the previous noisy sample x_t -> x_t-1
            primary_latents = self.scheduler.step(primary_pred, t, primary_latents, **extra_step_kwargs).prev_sample

            if t == timesteps[1] or t == timesteps[-1] and self.debug:
                print(f"Primary loop: primary_latents after timestep {t}", primary_latents.mean().cpu().item(), primary_latents.var().cpu().item()) # debug

        # 7. Re-prepare timesteps if primary loop is implemented
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler_secondary, num_inference_steps, device, timesteps_given, sigmas
        )

        # 8. Prepare latent variables (secondary)
        if secondary_latents is not None and not use_old_model:
            secondary_latents = dac_latents_normalize(secondary_latents, selection="secondary") if normalize_input_latents else secondary_latents
        elif secondary_latents is not None and use_old_model:
            secondary_latents = dac_latents_normalize_codebook_specific(secondary_latents, selection="secondary") if normalize_input_latents else secondary_latents
        
        secondary_latents = self.prepare_latents(
            batch_size * num_audios_per_prompt,
            num_channels_latents_secondary,
            sample_size,
            prompt_embeds_t5.dtype,
            device,
            generator,
            secondary_latents,
        )

        print("secondary_latents before loop", secondary_latents.mean().cpu().item(), secondary_latents.var().cpu().item()) # debug
        
        # 9. Denoising loop (secondary)
        self._num_timesteps = len(timesteps)
        for t in self.progress_bar(timesteps):
            # attach gt/sampled primary_latents as fixed conditioned
            secondary_latent_input = torch.cat([primary_latents, secondary_latents], dim=1) # [B, (1+K-1)d, L]
            
            # expand the latents if we are doing classifier free guidance
            secondary_latent_input = torch.cat([secondary_latent_input] * 2) if self.do_classifier_free_guidance else secondary_latent_input
            secondary_latent_input = self.scheduler_secondary.scale_model_input(secondary_latent_input, t)
            
            # predict the noise residual
            if not use_old_model:
                secondary_pred = self.model_secondary(
                    secondary_latent_input, t,
                    encoder_hidden_states=prompt_embeds_t5,
                    class_labels=class_labels,
                    timestep_cond=prompt_embeds_clap,
                    return_dict=True,
                ).sample # [B, (K-1)d, L]
            else:
                vec_cond = torch.cat([prompt_embeds_clap, prompt_embeds_clap * 0], dim = -1)
                timestep_tensor = torch.tensor(t).repeat(secondary_latent_input.shape[0]).to(device).long()
                secondary_pred = self.model_secondary.forward_with_cond_scale(
                    latent=secondary_latent_input,
                    timestep=timestep_tensor,
                    cond_scale=1.0,
                    vec_cond=vec_cond,
                    seq_conds=[None, prompt_embeds_t5.permute(0, 2, 1)]
                )
            
            # perform guidance
            if self.do_classifier_free_guidance:
                secondary_pred_uncond, secondary_pred_cond = secondary_pred.chunk(2)
                if self.debug:
                    print(f"Secondary loop: input primary_latents at timestep {t}", primary_latents.mean().cpu().item(), primary_latents.var().cpu().item())
                    print(f"Secondary loop: direct secondary_pred at timestep {t}", secondary_pred_cond.mean().cpu().item(), secondary_pred_cond.var().cpu().item()) # debug
                    print(f"Secondary loop: uncon secondary_pred at timestep {t}", secondary_pred_uncond.mean().cpu().item(), secondary_pred_uncond.var().cpu().item()) # debug
                secondary_pred = secondary_pred_uncond + self.guidance_scale * (secondary_pred_cond - secondary_pred_uncond)

            # debug
            # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
            #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            #     secondary_pred = rescale_noise_cfg(secondary_pred, secondary_pred_cond, guidance_rescale=self.guidance_rescale)

            if (t == timesteps[1] or t == timesteps[-1] or t == timesteps[10] or t == timesteps[-10]) and  self.debug:
                print(f"Secondary loop: secondary_pred at timestep {t}", secondary_pred.mean().cpu().item(), secondary_pred.var().cpu().item()) # debug
            
            # compute the previous noisy sample x_t -> x_t-1
            secondary_latents = self.scheduler_secondary.step(secondary_pred, t, secondary_latents, **extra_step_kwargs).prev_sample

            if (t == timesteps[1] or t == timesteps[-1] or t == timesteps[10] or t == timesteps[-10]) and  self.debug:
                print(f"Secondary loop: secondary_latents after timestep {t}", secondary_latents.mean().cpu().item(), secondary_latents.var().cpu().item()) # debug

        # 10. Get final latents and decode to audio
        if not use_old_model:
            latents = torch.cat([primary_latents, secondary_latents], dim=1) # [B, Kd, L]
            if self.debug:
                print(f"Generated latents", latents.mean().cpu().item(), latents.var().cpu().item()) # debug
            final_latents = dac_latents_denormalize(latents)
        else:
            primary_normalized = dac_latents_denormalize(primary_latents)
            secondary_normalized = dac_latents_denormalize_codebook_specific(secondary_latents, selection = "secondary")
            final_latents = torch.cat([primary_normalized, secondary_normalized], dim=1) # [B, Kd, L]

        if self.debug:
            print(f"Generated final primary latents", final_latents[:,:DAC_DIM_SINGLE].mean().cpu().item(), final_latents[:,:DAC_DIM_SINGLE].var().cpu().item()) # debug
            print(f"Generated final secondary latents", final_latents[:,DAC_DIM_SINGLE:].mean().cpu().item(), final_latents[:,DAC_DIM_SINGLE:].var().cpu().item()) # debug
            print(f"Generated final latents", final_latents.mean().cpu().item(), final_latents.var().cpu().item()) # debug

        print("Sampled latents shape", final_latents.shape) # debug
        z = self.dac_model.quantizer.from_latents(final_latents)[0] # [B, D, L]
        wav_sampled = self.dac_model.decode(z).squeeze(1) # [B, T]
        if torch.abs(wav_sampled).max() > 1:
            wav_sampled = wav_sampled / torch.abs(wav_sampled).max()

        if not return_dict:
            return (wav_sampled, final_latents)

        return AudioPipelineOutput(audios=wav_sampled, latents=final_latents)