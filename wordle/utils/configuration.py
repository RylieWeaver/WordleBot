# General
import json
import os

# Torch
import torch


def merge_defaults(defaults, config):
    for key, val in defaults.items():
        if key not in config:
            config[key] = val
        elif isinstance(val, dict) and isinstance(config[key], dict):
            merge_defaults(val, config[key])
    return config


def load_config(filename, default_filename="default_config.json"):
    # Load default config from the module directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(module_dir, default_filename)
    with open(default_path, 'r') as f:
        default_config = json.load(f)

    # Load user config from where the script runs
    user_config_path = os.path.abspath(filename)
    try:
        with open(user_config_path, 'r') as f:
            user_config = json.load(f)
    except FileNotFoundError:
        print(f"User config file {user_config_path} not found; using default config.")
        user_config = {}

    # Merge and return
    return merge_defaults(default_config, user_config)
