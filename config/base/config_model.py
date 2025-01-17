from config.base.attrdict import AttrDict

T5_MODELS_DIMS = {
    "t5-small": 512,
    "t5-base": 768,
    "t5-large": 1024,
    "t5-3b": 1024,
    "t5-11b": 1024,
    "google/flan-t5-small": 512,
    "google/flan-t5-base": 768,
    "google/flan-t5-large": 1024,
    "google/flan-t5-3b": 1024,
    "google/flan-t5-11b": 1024,
}

config_model_default = AttrDict(
    sample_size = 2240, # 2304, # 2320, # approx 27 sec dac
    in_channels_primary = 8,
    in_channels_secondary = 72,
    out_channels_primary = 8,
    out_channels_secondary = 64,
    down_block_types = (
        "CrossAttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "ResnetDownsampleBlock1D"
    ),
    mid_block_type = "UNetMidBlock1DCrossAttn",
    up_block_types = (
        "ResnetUpsampleBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "CrossAttnUpBlock1D"
    ),
    block_out_channels = (128, 128, 256, 256, 512, 512),

    down_block_types_secondary = (
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D"
    ),
    mid_block_type_secondary = "UNetMidBlock1D",
    up_block_types_secondary = (
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
    ),
    block_out_channels_secondary = (128, 256, 512, 1024),
    
    layers_per_block = 2,
    num_class_embeds = 24,
    class_embeddings_concat = False,
    encoder_hid_dim = 1024, # for flan-t5-large text conditioning
    time_embedding_dim = 512, # for CLAP text/audio conditioning
    t5_model_name = "google/flan-t5-large",
    
    sample_rate = 44100,
    num_codebooks = 9,
    codebook_size = 1024,
    codebook_dim = 8,
)

config_model_large = AttrDict(
    sample_size = 2240, # 2304, # 2320, # approx 27 sec dac
    in_channels_primary = 8,
    in_channels_secondary = 72,
    out_channels_primary = 8,
    out_channels_secondary = 64,
    down_block_types = (
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "ResnetDownsampleBlock1D"
    ),
    mid_block_type = "UNetMidBlock1DCrossAttn",
    up_block_types = (
        "ResnetUpsampleBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D"
    ),
    block_out_channels = (128, 256, 512, 512, 1024, 1024),

    down_block_types_secondary = (
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "AttnDownBlock1D",
    ),
    mid_block_type_secondary = "UNetMidBlock1D",
    up_block_types_secondary = (
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D",
    ),
    block_out_channels_secondary = (128, 256, 512, 512, 1024, 1024),
    
    layers_per_block = 2,
    num_class_embeds = 24,
    class_embeddings_concat = False,
    encoder_hid_dim = 1024, # for flan-t5-large text conditioning
    time_embedding_dim = 512, # for CLAP text/audio conditioning
    t5_model_name = "google/flan-t5-large",
    
    sample_rate = 44100,
    num_codebooks = 9,
    codebook_size = 1024,
    codebook_dim = 8,
)

config_model_large_old_secondary = AttrDict(
    sample_size = 2240, # 2304, # 2320, # approx 27 sec dac
    in_channels_primary = 8,
    in_channels_secondary = 72,
    out_channels_primary = 8,
    out_channels_secondary = 64,
    down_block_types = (
        "AttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "ResnetDownsampleBlock1D"
    ),
    mid_block_type = "UNetMidBlock1DCrossAttn",
    up_block_types = (
        "ResnetUpsampleBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "AttnUpBlock1D"
    ),
    block_out_channels = (128, 256, 512, 512, 1024, 1024),

    model_dim_in_old = 72,
    model_dim_out_old = 72,
    inner_dim_old = 576,
    head_dim_old = 128,
    dim_mults_old = (1, 1, 2, 2, 4),
    clap_dim = 512,
    meta_cond_dim = 512,
    chroma_dim = 12,
    
    layers_per_block = 2,
    num_class_embeds = 24,
    class_embeddings_concat = False,
    encoder_hid_dim = 1024, # for flan-t5-large text conditioning
    time_embedding_dim = 512, # for CLAP text/audio conditioning
    t5_model_name = "google/flan-t5-large",
    
    sample_rate = 44100,
    num_codebooks = 9,
    codebook_size = 1024,
    codebook_dim = 8,
)

config_model_tiny = AttrDict(
    sample_size = 256, # 2.97 sec dac
    in_channels_primary = 8,
    in_channels_secondary = 72,
    out_channels_primary = 8,
    out_channels_secondary = 64,
    down_block_types = (
        "CrossAttnDownBlock1D",
        "AttnDownBlock1D",
        "CrossAttnDownBlock1D",
        "ResnetDownsampleBlock1D"
    ),
    mid_block_type = "UNetMidBlock1DCrossAttn",
    up_block_types = (
        "ResnetUpsampleBlock1D",
        "CrossAttnUpBlock1D",
        "AttnUpBlock1D",
        "CrossAttnUpBlock1D"
    ),
    block_out_channels = (128, 128, 256, 512),
    layers_per_block = 2,
    num_class_embeds = 24, 
    class_embeddings_concat = False,
    encoder_hid_dim = 512, # for flan-t5-small text conditioning
    time_embedding_dim = 512, # for CLAP text/audio conditioning
    t5_model_name = "google/flan-t5-small",

    sample_rate = 44100,
    num_codebooks = 9,
    codebook_size = 1024,
    codebook_dim = 8,    
)