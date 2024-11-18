import os
import sys
import json
import importlib.util
from config.base.attrdict import AttrDict

def load_config_from_path(config_path):
    # determine config type according to input
    config_dir, config_filename = os.path.split(config_path)
    config_name, config_ext = os.path.splitext(config_filename)
    if config_ext.lower() == ".py":
        sys.path.append(config_dir)
        spec = importlib.util.spec_from_file_location("config", config_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        config = foo.config
    elif config_ext.lower() == ".json":
        with open(config_path) as data_file:
            config_dict = json.load(data_file)
        config = AttrDict(config_dict)
    else:
        raise ValueError("Input config_path should be a .py file or a .json file")

    return config