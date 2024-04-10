from abc import ABCMeta, abstractmethod
import inspect

import jax
import jax.numpy as jnp

from setup.settings import log_settings
from setup.parsers import (
    parse_verbosity_settings,
    parse_logging_settings,
    parse_plotting_settings,
    parse_directory_settings,
    parse_run_settings
)
from .optim import get_update
from .softadapt import softadapt


_DEFAULT_SEED: int = 0
_DEFAULT_ID: str = "generic_id"


class Model(metaclass=ABCMeta):
    """
    Base class that specific models such as PINNs and DeepONets inherit from.

    The main functionality on this level is to parse the settings specified in the JSON file.
    Additionally, abstract methods are defined.
    """

    def __init__(self, settings: dict, *args, **kwargs):
        """
        Initialize the model by calling the settings parsers.
        """
        
        self._parse_settings(settings)
        return

    def _parse_settings(self, settings: dict):
        """
        Parse various settings.
        """
        
        # Parse verbosity, seed and ID
        self._verbose = parse_verbosity_settings(settings.get("verbosity"))
        self._seed = settings.get("seed")
        if self._seed is None:
            if self._verbose.init:
                print(f"No seed specified in settings. Seed is set to {_DEFAULT_SEED}.")
            self._seed = _DEFAULT_SEED
        self._id = settings.get("id")
        if self._id is None:
            if self._verbose.init:
                print(f"No ID specified in settings. ID is set to '{_DEFAULT_ID}'.")
            self._id = _DEFAULT_ID
        
        # Set a key for use in other methods
        # Important: Remember to return a new _key as well, e.g.:
        #   self._key, key_for_some_task = jax.random.split(self._key)
        self._key = jax.random.PRNGKey(self._seed)

        # Parse more settings
        self._parse_directory_settings(settings["io"], self._id)
        self._parse_run_settings(settings["run"])
        self._parse_plotting_settings(settings["plotting"])
        self._parse_logging_settings(settings["logging"])
        
        if self.logging.do_logging:
            log_settings(settings, self.dir.log_dir, tensorboard=True, text_file=False)

        return

    def _parse_directory_settings(self, dir_settings: dict, id: str) -> None:
        """
        Parse settings related to file directories.
        """

        self.dir = parse_directory_settings(dir_settings, id)
        return

    def _parse_run_settings(self, run_settings: dict) -> None:
        """
        Parse settings related to the type of run.
        """

        self.train_settings, self.do_train = parse_run_settings(run_settings, run_type="train")
        self.eval_settings, self.do_eval = parse_run_settings(run_settings, run_type="eval")
        return
    
    def _parse_plotting_settings(self, plot_settings: dict) -> None:
        """
        Parse settings related to plotting.
        """
        
        self.sample_plots = parse_plotting_settings(plot_settings.get("sampling"))
        self.result_plots = parse_plotting_settings(plot_settings.get("results"))
        return
    
    def _parse_logging_settings(self, log_settings: dict) -> None:
        self.logging = parse_logging_settings(log_settings)
        return

    @abstractmethod
    def init_model(self) -> None:
        """
        This method should be implemented to initialize the model, e.g. the params.
        """
        pass

    @abstractmethod
    def forward(self):
        """
        The forward pass through a model.
        """
        pass

    def save_state(self):
        """
        Save model state. Call to a more general function write_model() ...
        """

        raise NotImplementedError("Method for saving model state is not implemented.")

    def load_state(self):
        """
        Load model state.
        """
        
        raise NotImplementedError("Method for loading model state is not implemented.")
    
    def _loss_terms(self) -> jax.Array:
        """
        Method for calculating loss terms. Loss terms may be defined
        in separate methods or calculated here.
        
        Must be overwritten by inheriting classes.
        """
        pass

    def _total_loss(self, *args, **kwargs):
        """
        This function sums up the loss terms return in the loss_terms
        method and passes through the auxillary output.
        """
        
        wloss, *aux = self._loss_terms(*args, **kwargs)
        return jnp.sum(wloss), tuple(aux)
    
    def _set_loss(self,
                  loss_term_fun_name: str) -> None:
        """
        (Stateful) method for setting loss function.
        """

        # Get default ("unweighted") update function
        fun = getattr(self, loss_term_fun_name)
        
        # Check if SoftAdapt should be used
        if self.train_settings.update_scheme == "softadapt":
            fun = softadapt(**self.train_settings.update_kwargs)(fun)
        
        # Set loss term function
        self._loss_terms = fun
        return

    def _set_update(self,
                    loss_fun_name: str = "_total_loss",
                    optimizer_name: str = "optimizer"
                    ) -> None:
        """
        Method for setting update function.

        Currently supported:
            The usual unweighted update
            The SoftAdapt update

        Args:
            loss_fun_name:   Name of function computing the total loss.
            optimizer_name:  Name of the optimizer used.
        """

        loss_fun = getattr(self, loss_fun_name)
        optimizer = getattr(self, optimizer_name)
        
        self.update = get_update(loss_fun,
                                 optimizer,
                                 self.train_settings.update_scheme,
                                 self.train_settings.jitted_update)
        return
 
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