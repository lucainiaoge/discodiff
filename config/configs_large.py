from config.base.attrdict import AttrDict
from config.base.config_training import config_training_default
from config.base.config_diffusion import config_diffusion_ddpm
from config.base.config_model import config_model_large

# default configuration
config = AttrDict()
config = config.override(config_diffusion_ddpm)
config = config.override(config_training_default)
config = config.override(config_model_large)

# custom configuration: add/modify the base configs, and do a similar process as shown above
# do ensure that config_diffusion_custom is defined in config.base.config_diffusion and so on for training and model configs
# config = AttrDict()
# config = config.override(config_diffusion_custom)
# config = config.override(config_training_custom)
# config = config.override(config_model_custom)
