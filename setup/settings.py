from dataclasses import dataclass
from collections.abc import Callable
import pathlib
import json

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from torch.utils.tensorboard import SummaryWriter


class SettingsInterpretationError(Exception):
    pass


class SettingsNotSupportedError(Exception):
    pass


class Settings:
    pass


@dataclass
class SupportedCustomInitializers:
    """
    Besides these, all functions from the
    flax.linen.initializers module are supported.
    """
    pass


@dataclass
class SupportedCustomOptimizerSchedules:
    """
    Besides these, all functions from the
    optax.schedules module are supported.
    """
    pass


@dataclass
class SupportedActivations:
    tanh: Callable = nn.tanh
    sigmoid: Callable = nn.sigmoid
    silu: Callable = nn.silu
    swish: Callable = nn.silu
    sin: Callable = jax.jit(jnp.sin)
    cos: Callable = jax.jit(jnp.cos)


@dataclass
class SupportedOptimizers:
    adam: Callable = optax.adam
    adamw: Callable = optax.adamw
    set_to_zero: Callable = optax.set_to_zero


@dataclass
class SupportedEquations:
    """
    Class for supported equations. Not in use yet.
    """
    pass


@dataclass
class SupportedSamplingDistributions:
    uniform: Callable = jax.random.uniform


@dataclass
class VerbositySettings(Settings):
    init: bool = True
    training: bool = True
    evaluation: bool = True
    plotting: bool = True
    sampling: bool = True


@dataclass
class LoggingSettings(Settings):
    do_logging: bool = True
    log_every: int | None = None
    print_every: int | None = None


@dataclass
class DirectorySettings(Settings):
    base_dir: pathlib.Path
    figure_dir: pathlib.Path | None = None
    model_dir: pathlib.Path | None = None
    image_dir: pathlib.Path | None = None
    log_dir: pathlib.Path | None = None


@dataclass
class TrainingSettings(Settings):
    sampling: dict
    iterations: int = 1000
    optimizer: Callable = SupportedOptimizers.adam
    update_scheme: str = "unweighted"
    update_kwargs: dict | None = None
    learning_rate: float = 1e-3
    batch_size: int | None = None
    decay_rate: float | None = None
    decay_steps: int | None = None
    transfer_learning: bool = False
    resampling: dict | None = None
    jitted_update: bool = True


@dataclass
class EvaluationSettings(Settings):
    sampling: dict
    error_metric: str = "L2-rel"
    transfer_learning: bool = False
    pass


@dataclass
class PlottingSettings(Settings):
    do_plots: bool = False
    plot_every: int | None = None
    overwrite: bool = False
    file_extension: str = "png"
    kwargs: dict | None = None

@dataclass
class SoftAdaptSettings(Settings):
    order: int = 1
    beta: float = 0.1
    normalized: bool = False
    loss_weighted: bool = False
    delta_time: float | None = None
    shift_by_max_val: bool = True


@dataclass
class MLPSettings(Settings):
    name: str = "MLP"
    input_dim: int = 1
    output_dim: int = 1
    hidden_dims: int | list[int] = 32
    activation: Callable | list[Callable] = SupportedActivations.tanh
    initialization: Callable | list[Callable] = nn.initializers.glorot_normal


def log_settings(settings_dict: dict,
                 log_dir: pathlib.Path,
                 *,
                 tensorboard: bool = False,
                 text_file: bool = False
                 ) -> None:
    """
    Logs JSON file of settings in Tensorboard and/or a text file.
    """

    if text_file:
        #TODO
        raise NotImplementedError("Logging to a text file is not supported yet.")

    # Function for JSON formatting
    def pretty_json(hp):
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))
    
    # os.system("rm -rf " + settings["io"]["log_dir"]+"/"+settings["id"]+"/*")
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text("settings.json", pretty_json(settings_dict))
    writer.close()
    return


def settings2dict(settings: Settings) -> dict:
    return settings.__dict__