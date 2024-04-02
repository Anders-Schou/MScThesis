from abc import ABCMeta, abstractmethod
import pathlib
import inspect

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax.training import checkpoints
import orbax.checkpoint as ocp

from setup.settings import DirectorySettings, TrainingSettings, EvaluationSettings
from setup.parsers import parse_training_settings, parse_evaluation_settings, parse_directory_settings


class Model(metaclass=ABCMeta):
    """
    Base class that specific models such as PINNs and DeepONets inherit from.
    """
    def __init__(self, settings: dict, *args, **kwargs):
        """
        Initialize the model by calling the settings parsers.
        """
        # Parse various settings such as training/evaluation, plotting and IO
        self._parse_settings(settings)
        return

    def _parse_settings(self, settings: dict):
        """
        Parse various settings.
        """
        # Parse directory settings
        self._parse_directory_settings(settings["io"], settings["id"])
        # Parse run settings
        self._parse_run_settings(settings["run"])
        # Parse plot settings
        self._parse_plot_settings(settings["plotting"])
        return

    def _parse_directory_settings(self, dir_settings: dict, id: str) -> None:
        """
        Parse settings related to file directories.
        """
        dir = DirectorySettings(**jtu.tree_map(pathlib.Path, dir_settings))
        self.dir = parse_directory_settings(dir, id)
        for d in self.dir.__dict__.values():
            d.mkdir(parents=True, exist_ok=True)
        return

    def _parse_run_settings(self, run_settings: dict) -> None:
        """
        Parse settings related to the type of run.
        """
        self.do_train = "train" in run_settings.keys()
        self.do_eval = "eval" in run_settings.keys()
        self.train_settings = parse_training_settings(run_settings["train"]) if self.do_train else None
        self.eval_settings = parse_evaluation_settings(run_settings["eval"]) if self.do_eval else None
        return
    
    def _parse_plot_settings(self, plot_settings: dict) -> None:
        """
        Parse settings related to plotting.
        """
        self.do_sample_data_plots = plot_settings["do_sample_data_plots"]
        return

    @abstractmethod
    def forward(self):
        """
        The forward pass through a model.
        """
        pass

    def save_state(self):
        """
        Save model state. Call to write_model() ...
        """
        pass

    def load_state(self):
        """
        Load model state.
        """
        pass
    
    @abstractmethod
    def loss_terms(self) -> jax.Array:
        """
        Method for calculating loss terms. Loss terms may be defined
        in separate methods or calculated here.
        
        Must be overwritten by inheriting classes.
        """
        pass

    def total_loss(self, *args, **kwargs):
        """
        This function sums up the loss terms return in the loss_terms
        method and passes through the auxillary output.
        """
        wloss, *aux = self.loss_terms(*args, **kwargs)
        return jnp.sum(wloss), tuple(aux)
    
    @abstractmethod
    def update(self):
        """
        Method for updating parameters of a model during training.

        Must be overwritten by inheriting classes.
        """
        pass

    @abstractmethod
    def train(self):
        """
        Method for training a model. Typically a loop structure with
        an update procedure in the loop. The update should be defined
        in a separate method so it can be JIT'ed easily.
        
        Must be overwritten by inheriting classes.
        """
        pass

    @abstractmethod
    def eval(self):
        """
        Method for evaluating a model.
        
        Must be overwritten by inheriting classes.
        """
        pass

    def __str__(self):
        """
        The string representation of the model.

        Prints out the methods/attributes and the documentation if any.
        """
        
        s = f"\n\nModel '{self.__class__.__name__}' with the following methods:\n\n\n\n"
        # print(dir(self))
        for m in dir(self):
            attr = getattr(self, m)
            if m.startswith("_") or not inspect.ismethod(attr):
                continue
            s += m
            s += ":\n"
            docstr = attr.__doc__
            s += docstr if docstr is not None else "\n\tNo documentation.\n"
            s += "\n\n"
        
        s += "\n\n\n\n\n... and the following attributes:\n\n\n\n"

        for m in dir(self):
            attr = getattr(self, m)
            if m.startswith("_") or inspect.ismethod(attr):
                continue
            s += m
            # s += ":\n"
            # docstr = attr.__doc__
            # s += docstr if docstr is not None else "\n\tNo documentation.\n"
            s += "\n\n"

        return s