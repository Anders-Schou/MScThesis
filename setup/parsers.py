import json
import argparse
from collections.abc import Callable

import jax
import jax.numpy as jnp
import flax.linen as nn

from setup.settings import (settings2dict, MLPSettings, TrainingSettings,
                            SupportedActivations, SupportedOptimizers,
                            SettingsInterpretationError, SettingsNotSupportedError)


def load_json(path: str) -> dict:
    try:
        f = open(path, "r")
    except FileNotFoundError:
        print(f"Could not find settings file: '{path}'")
        exit()
    j = json.loads(f.read())
    f.close()
    return j


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    args = parser.parse_args()
    json_dict = load_json(args.settings)
    return json_dict


def parse_loss_type(loss_str: str) -> Callable:
    """
    Based on input string, this function returns a function
    for calculating the outer loss, e.g. a mean-squared error.
    """
    from models.loss import mse
    
    loss_str = loss_str.lower()
    if loss_str == "mse":
        return mse
    raise ValueError(f"Loss of type '{loss_str}' is not supported.")


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
    settings_dict = settings_dict.copy()

    # Get default settings
    settings = MLPSettings()

    # input_dim
    if settings_dict.get("input_dim") is not None:
        check_network_dims(settings_dict["input_dim"], "input_dim")
        settings.input_dim = settings_dict["input_dim"]
    
    # output_dim
    if settings_dict.get("output_dim") is not None:
        check_network_dims(settings_dict["output_dim"], "output_dim")
        settings.output_dim = settings_dict["output_dim"]
    
    # hidden_dims
    if settings_dict.get("hidden_dims") is not None:
        if isinstance(settings_dict["hidden_dims"], int):
           settings_dict["hidden_dims"] = [settings_dict["hidden_dims"]]
        for hidden_dim in settings_dict["hidden_dims"]:
            check_network_dims(hidden_dim, "hidden_dims")
        settings.hidden_dims = settings_dict["hidden_dims"]
        num_hidden = len(settings.hidden_dims)

    # activation
    if settings_dict.get("activation") is not None:
        if isinstance(settings_dict["activation"], list):
            if len(settings_dict["activation"]) != num_hidden:
                raise SettingsInterpretationError(
                    "List of activation functions does not correspond to number of hidden layers.")
            if not all([isinstance(act, str) for act in settings_dict["activation"]]):
                raise SettingsInterpretationError(
                    "List of activation functions must be strings.")
            settings_dict["activation"] = [convert_activation(act) for act in settings_dict["activation"]]
        elif isinstance(settings_dict["activation"], str):
            settings_dict["activation"] = [convert_activation(settings_dict["activation"])] * num_hidden
        else:
            raise SettingsInterpretationError(
                    "Wrong type for activation setting.")
        settings.activation = settings_dict["activation"]
    else:
        settings.activation = [settings.activation] * num_hidden
    
    # initialization
    if settings_dict.get("initialization") is not None:
        if isinstance(settings_dict["initialization"], list):
            if len(settings_dict["initialization"]) != num_hidden:
                raise SettingsInterpretationError(
                    "List of initialization functions does not correspond to number of hidden layers.")
            if not all([isinstance(init, str) for init in settings_dict["initialization"]]):
                raise SettingsInterpretationError(
                    "List of initialization functions must be strings.")
            settings_dict["initialization"] = [convert_initialization(init) for init in settings_dict["initialization"]]
        elif isinstance(settings_dict["initialization"], str):
            settings_dict["initialization"] = [convert_initialization(settings_dict["initialization"])] * (num_hidden+1)
        else:
            raise SettingsInterpretationError(
                    "Wrong type for initialization setting.")
        settings.initialization = settings_dict["initialization"]
    else:
        settings.initialization = [settings.initialization] * (num_hidden+1)
    
    return settings2dict(settings)


def parse_training_settings(settings_dict: dict) -> dict:
    """
    Parses settings specified in dictionary.
    Valid settings include those in the default
    settings class:

        iterations: int = 1000
        optimizer: str = "adam"
        learning_rate: float = 1e-3
        batch_size: int | None = None
        decay_rate: float | None = None
        decay_steps: int | None = None
        transfer_learning: bool = False

    Raises exception if a setting is unknown,
    or if theres a mismatch in length of lists
    of activations, initializations and number
    of neurons in hidden layers.
    """
    settings_dict = settings_dict.copy()

    # Get default settings
    settings = TrainingSettings()

    # Load settings from dictionary into settings class
    for key, value in settings_dict.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            raise SettingsInterpretationError(
                f"Error: '{key}' is not a valid setting.")
    
    settings.optimizer = convert_optimizer(settings.optimizer)
    
    return settings2dict(settings)


def convert_activation(act_str: str) -> Callable:
    """
    Converts activation string to activation function (callable),
    based on the supported activations.
    """
    try:
        act_fun = getattr(SupportedActivations, act_str)
    except Exception as err:
        raise SettingsNotSupportedError(
            f"Activation function '{act_str} is not supported.") from err
    return act_fun


def convert_initialization(init_str: list[str]) -> list[Callable]:
    """
    Converts list of initialization strings to list of initialization
    functions (callables), based on the supported initialization.
    """
    try:
        init_fun = getattr(nn.initializers, init_str)
    except Exception as err:
        raise SettingsNotSupportedError(
            f"Initialization '{init_str} is not supported.") from err
    return init_fun


def convert_optimizer(opt_str: str):
    """
    Converts optimizer string to optimizer object.
    """
    try:
        opt_fun = getattr(SupportedOptimizers, opt_str)
    except Exception as err:
        raise SettingsNotSupportedError(
            f"Optimizer '{opt_str}' is not supported.") from err
    return opt_fun


def convert_sampling_distribution(dist_str: str) -> Callable:
    """
    Converts sampling distribution string to function (callable).
    """
    try:
        dist_fun = getattr(jax.random, dist_str)
    except Exception as err:
        raise SettingsNotSupportedError(
            f"Sampling distribution '{dist_str}' is not supported.") from err
    return dist_fun


def check_network_dims(option, name) -> None:
    if not isinstance(option, int):
        raise SettingsInterpretationError(f"Option '{name}' must be an integer.")
    if not option >= 1:
        raise SettingsInterpretationError(f"Option '{name}' must be positive.")
    return