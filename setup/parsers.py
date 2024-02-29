import json
import argparse
from collections.abc import Callable

import jax
import flax.linen as nn

from setup.settings import (settings2dict, MLPSettings, SupportedActivations,
    SettingsInterpretationError, SettingsNotSupportedError)


def load_json(path: str):
    f = open(path, "r")
    j = json.loads(f.read())
    f.close()
    return j


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    args = parser.parse_args()
    json_dict = load_json(args.settings)
    return json_dict


def parse_MLP_settings(settings_dict: dict) -> dict:
    """
    Parses settings specified in dictionary.
    Valid settings include those in the default
    settings class:

        input_dim: int = 1
        output_dim: int = 1
        hidden_dims: int | list[int] = 32
        activation: str | list[str] = "tanh"
        initialization: str | list[str] = "glorot_normal"

    Raises exception if a setting is unknown,
    or if theres a mismatch in length of lists
    of activations, initializations and number
    of neurons in hidden layers.
    """
    # Get default settings
    settings = MLPSettings()

    # Load settings from dictionary into settings class
    for key, value in settings_dict.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise SettingsInterpretationError(
                f"Error: '{key}' is not a valid setting.")
    
    # Ensure hidden_dims is a list
    if isinstance(settings.hidden_dims, int):
        settings.hidden_dims = [settings.hidden_dims]
    num_hidden = len(settings.hidden_dims)

    # List of activations should have same length as hidden_dims
    if isinstance(settings.activation, list):
        if len(settings.activation) != num_hidden:
            raise SettingsInterpretationError(
                 "List of activation functions does not correspond to number of hidden layers.")
    else:
        settings.activation = [settings.activation] * num_hidden
    for a in settings.activation:
        if not isinstance(a, str):
            raise SettingsInterpretationError(
                "Wrong type for initialization setting.")
    settings.activation = convert_activation(settings.activation)
    
    # List of initializations should have length = len(hidden_dims) + 1
    if isinstance(settings.initialization, list):
        if len(settings.initialization) != num_hidden + 1:
            raise SettingsInterpretationError(
                 "List of weight initializations does not correspond to number of hidden layers.")
    else:
        settings.initialization = [settings.initialization] * (num_hidden + 1)
    for i in settings.initialization:
        if not isinstance(i, str):
            raise SettingsInterpretationError(
                "Wrong type for initialization setting.")
    settings.initialization = convert_initialization(settings.initialization)
    
    return settings2dict(settings)


def convert_activation(act_str: list[str]) -> list[Callable]:
    supported_activations = SupportedActivations()
    act_fun = []
    for a in act_str:
        try:
            act_fun.append(getattr(supported_activations, a))
        except Exception as err:
            raise SettingsNotSupportedError(f"Activation function '{a} is not supported.") from err
    return act_fun


def convert_initialization(init_str: list[str]) -> list[Callable]:
    init_fun = []
    for i in init_str:
        try:
            init_fun.append(getattr(nn.initializers, i))
        except Exception as err:
            raise SettingsNotSupportedError(f"Initialization '{i} is not supported.") from err
    return init_fun


def convert_sampling_distribution(dist_str: str) -> Callable:
    try:
        dist_fun = getattr(jax.random, dist_str)
    except Exception as err:
        raise SettingsNotSupportedError(f"Sampling distribution {dist_str} is not supported.") from err
    return dist_fun