from config.base.attrdict import AttrDict

config_diffusion_default = AttrDict(
    num_train_timesteps = 1000,
    num_inference_timesteps = 50,
    prediction_type = 'v_prediction', # choose from ['epsilon', 'sample', 'v_prediction']
    cfg_drop_prob = 0.25,
    cfg_scale = 7.5,
    cfg_rescale = 0.1,
)
