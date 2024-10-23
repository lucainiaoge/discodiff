from config.base.attrdict import AttrDict
from config.base.config_training import config_training_default
from config.base.config_diffusion import config_diffusion_default
from config.base.config_model import config_model_tiny

# tiny configuration
config = AttrDict()
config = config.override(config_diffusion_default)
config = config.override(config_training_default)
config = config.override(config_model_tiny)
