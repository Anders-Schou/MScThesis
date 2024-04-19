import json
import argparse
from collections.abc import Callable
import pathlib
import shutil

import jax
import jax.tree_util as jtu
import flax.linen as nn
import optax

from setup.settings import (
    VerbositySettings, 
    LoggingSettings,
    MLPSettings,
    ResNetBlockSettings,
    TrainingSettings,
    EvaluationSettings,
    PlottingSettings,
    DirectorySettings,
    SoftAdaptSettings,
    WeightedSettings,
    UnweightedSettings,
    SupportedActivations,
    SupportedOptimizers,
    SupportedCustomOptimizerSchedules,
    SupportedCustomInitializers,
    SettingsInterpretationError,
    SettingsNotSupportedError,
    settings2dict
)


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


def parse_verbosity_settings(settings_dict: dict | None = None):
    if settings_dict is None:
        return VerbositySettings()
    if settings_dict.get("all") is not None:
        return VerbositySettings()
    return VerbositySettings(**settings_dict)


def parse_logging_settings(settings_dict: dict):
    return LoggingSettings(**settings_dict)


def parse_plotting_settings(settings_dict: dict | None) -> PlottingSettings:
    """
    Parses settings related to plotting.
    """

    if settings_dict is None:
        return PlottingSettings()
    return PlottingSettings(**settings_dict)


def parse_run_settings(settings_dict: dict, run_type: str
                       ) -> tuple[TrainingSettings | EvaluationSettings, bool]:
    """
    Parses settings related to running the model.
    """

    settings_dict = settings_dict.copy()

    if run_type == "train":
        if "train" in settings_dict.keys():
            train_settings = parse_training_settings(settings_dict["train"])
            do_train = True
        if "do_train" in settings_dict.keys():
            do_train = settings_dict["do_train"]
        return train_settings, do_train

    if run_type == "eval":
        if "eval" in settings_dict.keys():
            eval_settings = parse_evaluation_settings(settings_dict["eval"])
            do_eval = True
        if "do_eval" in settings_dict.keys():
            do_train = settings_dict["do_eval"]
        return eval_settings, do_eval

    raise ValueError(f"Invalid run type: '{run_type}'.")


def parse_MLP_settings(settings_dict: dict) -> dict:
    """
    Parses settings specified in dictionary.
    Valid settings include those in the default
    settings class:

        name: str = "MLP"
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

    # name
    if settings_dict.get("name") is not None:
        settings.name = str(settings_dict["name"])

    # input_dim
    if settings_dict.get("input_dim") is not None:
        check_pos_int(settings_dict["input_dim"], "input_dim")
        settings.input_dim = settings_dict["input_dim"]
    
    # output_dim
    if settings_dict.get("output_dim") is not None:
        check_pos_int(settings_dict["output_dim"], "output_dim")
        settings.output_dim = settings_dict["output_dim"]
    
    # hidden_dims
    if settings_dict.get("hidden_dims") is not None:
        if isinstance(settings_dict["hidden_dims"], int):
           settings_dict["hidden_dims"] = [settings_dict["hidden_dims"]]
        for hidden_dim in settings_dict["hidden_dims"]:
            check_pos_int(hidden_dim, "hidden_dims")
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


def parse_ResNetBlock_settings(settings_dict: dict) -> dict:
    """
    Parses settings specified in dictionary.
    Valid settings include those in the default
    settings class:

        name: str = "MLP"
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
    settings = ResNetBlockSettings()

    # name
    if settings_dict.get("name") is not None:
        settings.name = str(settings_dict["name"])
    
    # input_dim
    if settings_dict.get("input_dim") is not None:
        check_pos_int(settings_dict["input_dim"], "input_dim")
        settings.input_dim = settings_dict["input_dim"]
    
    # output_dim
    if settings_dict.get("output_dim") is not None:
        check_pos_int(settings_dict["output_dim"], "output_dim")
        settings.output_dim = settings_dict["output_dim"]
    
    # hidden_dims
    if settings_dict.get("hidden_dims") is not None:
        if isinstance(settings_dict["hidden_dims"], int):
           settings_dict["hidden_dims"] = [settings_dict["hidden_dims"]]
        for hidden_dim in settings_dict["hidden_dims"]:
            check_pos_int(hidden_dim, "hidden_dims")
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
    
    # pre_act
    if settings_dict.get("pre_act") is not None:
        if isinstance(settings_dict["pre_act"], list):
            settings_dict["pre_act"] = settings_dict["pre_act"][0]
        if not isinstance(settings_dict["pre_act"], str):
            raise SettingsInterpretationError("Cannot interpret pre-activation function. Input must be a string")
        settings_dict["pre_act"] = convert_activation(settings_dict["pre_act"])
    
    # pre_act
    if settings_dict.get("post_act") is not None:
        if isinstance(settings_dict["post_act"], list):
            settings_dict["post_act"] = settings_dict["post_act"][0]
        if not isinstance(settings_dict["post_act"], str):
            raise SettingsInterpretationError("Cannot interpret post-activation function. Input must be a string")
        settings_dict["post_act"] = convert_activation(settings_dict["post_act"])

    # shortcut_init
    if settings_dict.get("shortcut_init") is not None:
        if isinstance(settings_dict["shortcut_init"], list):
            settings_dict["shortcut_init"] = settings_dict["shortcut_init"][0]
        if not isinstance(settings_dict["shortcut_init"], str):
            raise SettingsInterpretationError("Cannot interpret shortcut initialization function. Input must be a string")
        settings_dict["shortcut_init"] = convert_initialization(settings_dict["shortcut_init"])
    return settings2dict(settings)


def parse_training_settings(settings_dict: dict) -> TrainingSettings:
    """
    Parses settings related to training.

    Returns a TrainingSettings object.
    """

    settings_dict = settings_dict.copy()

    # Get default settings (provide only required argument(s))
    settings = TrainingSettings(settings_dict["sampling"])

    # iterations
    if settings_dict.get("iterations") is not None:
        check_pos_int(settings_dict["iterations"], "iterations", strict=False)
        settings.iterations = settings_dict["iterations"]
    
    # optimizer
    if settings_dict.get("optimizer") is not None:
        settings.optimizer = convert_optimizer(settings_dict["optimizer"])
    
    # update_scheme
    if settings_dict.get("update_scheme") is not None:
        settings.update_scheme = settings_dict["update_scheme"].lower()
    
    # update_kwargs
    if settings_dict.get("update_kwargs") is not None:
        settings.update_kwargs = {}

        if settings.update_scheme == "softadapt":
            try:
                settings.update_kwargs["softadapt"] = settings2dict(
                    SoftAdaptSettings(**settings_dict["update_kwargs"]["softadapt"])
                )
            except KeyError:
                settings.update_kwargs["softadapt"] = settings2dict(SoftAdaptSettings())
        
        elif settings.update_scheme == "weighted":
            try:
                settings.update_kwargs["weighted"] = settings2dict(
                    WeightedSettings(**settings_dict["update_kwargs"]["weighted"])
                )
            except (KeyError, TypeError) as e:
                raise KeyError("Using the weighted update requires a settings dict "
                               "with the field 'weights'.") from e
        
        else:
            try:
                settings.update_kwargs["unweighted"] = settings2dict(
                    UnweightedSettings(**settings_dict["update_kwargs"]["unweighted"])
                )
            except KeyError:
                settings.update_kwargs["unweighted"] = settings2dict(UnweightedSettings())
    
    # learning_rate
    if settings_dict.get("learning_rate") is not None:
        check_pos(settings_dict["learning_rate"], "learning_rate")
        settings.learning_rate = settings_dict["learning_rate"]
    
    # batch_size
    if settings_dict.get("batch_size") is not None:
        check_pos_int(settings_dict["batch_size"], "batch_size")
        settings.batch_size = settings_dict["batch_size"]
    
    # decay_rate
    if settings_dict.get("decay_rate") is not None:
        check_pos(settings_dict["decay_rate"], "decay_rate")
        settings.decay_rate = settings_dict["decay_rate"]

    # decay_steps
    if settings_dict.get("decay_steps") is not None:
        check_pos_int(settings_dict["decay_steps"], "decay_steps")
        settings.decay_steps = settings_dict["decay_steps"]
    
    # transfer_learning
    if settings_dict.get("transfer_learning") is not None:
        settings.transfer_learning = settings_dict["transfer_learning"]
    
    # resampling
    if settings_dict.get("resampling") is not None:
        settings.resampling = settings_dict["resampling"]

    # jitted_update
    if settings_dict.get("jitted_update") is not None:
        settings.jitted_update = settings_dict["jitted_update"]
    
    return settings


def parse_evaluation_settings(settings_dict: dict) -> EvaluationSettings:
    """
    Parses settings related to evaluation.

    Returns an EvaluationSettings object.
    """
    
    settings_dict = settings_dict.copy()

    # Get default settings (provide only required argument(s))
    settings = EvaluationSettings(settings_dict["sampling"])

    # error_metric
    if settings_dict.get("error_metric") is not None:
        settings.error_metric = settings_dict["error_metric"]
    
    # transfer_learning
    if settings_dict.get("transfer_learning") is not None:
        settings.transfer_learning = settings_dict["transfer_learning"]

    return settings


def parse_directory_settings(settings_dict: dict,
                             id: str,
                             log_str: str | None = None
                             ) -> DirectorySettings:
    """
    Parses settings related to file directories.

    Only base_dir is required in the settings_dict.
    Other dirs will be generated from base_dir if not specified.

    Returns a DirectorySettings object.
    """

    if log_str is None:
        # Files to remove in log_dir
        log_str = "*tfevents*"

    # Directory class
    dir = DirectorySettings(**jtu.tree_map(
        lambda path_str: pathlib.Path(path_str), settings_dict)
        )

    # Directory for storing figures
    if dir.figure_dir is None:
        setattr(dir, "figure_dir", dir.base_dir / "figures")
    dir.figure_dir = dir.figure_dir / id
    dir.figure_dir.mkdir(parents=True, exist_ok=True)

    # Directory for storing models
    if dir.model_dir is None:
        setattr(dir, "model_dir", dir.base_dir / "models")
    dir.model_dir = dir.model_dir / id
    dir.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory for storing images
    if dir.image_dir is None:
        setattr(dir, "image_dir", dir.base_dir / "images")
    dir.image_dir = dir.image_dir / id
    dir.image_dir.mkdir(parents=True, exist_ok=True)

    # Directory for storing log files (e.g. Tensorboard files or text-based log files)
    if dir.log_dir is None:
        setattr(dir, "log_dir", dir.base_dir / "logs")
    dir.log_dir = dir.log_dir / id
    shutil.rmtree(dir.log_dir / log_str, ignore_errors=True) # Remove current log_dir if it exists
    dir.log_dir.mkdir(parents=True, exist_ok=True)

    return dir


def convert_activation(act_str: str) -> Callable:
    """
    Converts activation string to activation function (callable),
    based on the supported activations.
    """
    try:
        act_fun = getattr(SupportedActivations, act_str)
    except AttributeError as err:
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
    except AttributeError:
        try:
            init_fun = getattr(SupportedCustomInitializers, init_str)
        except AttributeError as err:
            raise SettingsNotSupportedError(
                f"Initialization '{init_str} is not supported.") from err
    return init_fun


def convert_optimizer(opt_str: str):
    """
    Converts optimizer string to optimizer object.
    """
    try:
        opt_fun = getattr(SupportedOptimizers, opt_str)
    except AttributeError as err:
        raise SettingsNotSupportedError(
            f"Optimizer '{opt_str}' is not supported.") from err
    return opt_fun


def convert_schedule(schedule_str: str):
    """
    Converts optimizer schedule string to optimizer schedule object.
    """
    try:
        schedule_fun = getattr(optax.schedules, schedule_str)
    except AttributeError:
        try:
            schedule_fun = getattr(SupportedCustomOptimizerSchedules, schedule_str)
        except AttributeError as err:
            raise SettingsNotSupportedError(
                f"Optimizer schedule '{schedule_str}' is not supported.") from err
    return schedule_fun


def convert_sampling_distribution(dist_str: str) -> Callable:
    """
    Converts sampling distribution string to function (callable).
    """
    try:
        dist_fun = getattr(jax.random, dist_str)
    except AttributeError as err:
        raise SettingsNotSupportedError(
            f"Sampling distribution '{dist_str}' is not supported.") from err
    return dist_fun


def check_pos_int(option, name, strict = True) -> None:
    """
    Ensures input is integer and positive.
    """
    check_int(option, name)
    check_pos(option, name, strict=strict)
    return


def check_int(option, name):
    """
    Ensures input is integer.
    """

    if not isinstance(option, int):
        raise SettingsInterpretationError(f"Option '{name}' must be an integer.")
    return


def check_pos(option, name, strict = True):
    """
    Ensures input is positive.
    """

    satisfied = (option >= 0)
    if strict:
        satisfied = (option > 0)

    if not satisfied > 0:
        raise SettingsInterpretationError(f"Option '{name}' must be positive.")
    return