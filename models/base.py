from abc import ABCMeta, abstractmethod
from collections.abc import Callable
import inspect

import jax
import jax.numpy as jnp
import optax

from setup.settings import (
    ModelNotInitializedError,
    log_settings,
    DefaultSettings,
    SoftAdaptSettings,
    WeightedSettings,
    UnweightedSettings
)
from setup.parsers import (
    parse_verbosity_settings,
    parse_logging_settings,
    parse_plotting_settings,
    parse_directory_settings,
    parse_run_settings
)
from .optim import get_update
from .loss import softadapt, weighted, unweighted


class Model(metaclass=ABCMeta):
    """
    Base class that specific models such as PINNs and DeepONets inherit from.

    The main functionality on this level is to parse the settings specified in the JSON file.
    Additionally, abstract methods are defined.
    """
    params: optax.Params
    train_points: dict
    train_true_val: dict | None
    eval_points: dict
    eval_true_val: dict | None

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
                print(f"No seed specified in settings. Seed is set to {DefaultSettings.SEED}.")
            self._seed = DefaultSettings.SEED
        self._id = settings.get("id")
        if self._id is None:
            if self._verbose.init:
                print(f"No ID specified in settings. ID is set to '{DefaultSettings.ID}'.")
            self._id = DefaultSettings.ID
        
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
        
        if settings.get("description"):
            with open(self.dir.log_dir / "description.txt", "a+") as file:
                file.writelines([settings["description"], "\n\n"])

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
        plot_settings.pop("sampling")
        plot_settings.pop("results")
        self.plot_settings = plot_settings
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
            self._loss_terms = softadapt(SoftAdaptSettings(
                **self.train_settings.update_kwargs["softadapt"]))(fun)
        elif self.train_settings.update_scheme == "weighted":
            self._loss_terms = weighted(WeightedSettings(
                **self.train_settings.update_kwargs["weighted"]))(fun)
        else:
            self._loss_terms = unweighted(UnweightedSettings(
                **self.train_settings.update_kwargs["unweighted"]))(fun)

        return
    
    def _init_prevlosses(self,
                         loss_term_fun: Callable[..., jax.Array],
                         update_key: int | None = None,
                         prevlosses: jax.Array | None = None
                         ) -> None:
        """
        Method for initializing array of previous losses.
        If prevlosses is None, the method sets prevlosses
        to an array determined by the loss type.
        """
        
        if not self._initialized():
            raise ModelNotInitializedError("The model has not been initialized properly.")
        
        # Calculate initial loss
        init_loss = loss_term_fun(self.params, self.train_points, 
                                  true_val=self.train_true_val, update_key=update_key)
        
        if len(init_loss.shape) == 0:
            loss_shape = 1
        else:
            loss_shape = init_loss.shape[0]

        # Check which loss to use
        if self.train_settings.update_scheme == "softadapt":
            numrows = self.train_settings.update_kwargs["softadapt"]["order"] + 1
            if prevlosses is None or prevlosses.shape[0] < numrows:
                self.prevlosses = jnp.tile(init_loss, numrows).reshape((numrows, loss_shape)) \
                    * jnp.linspace(numrows, 1, numrows).reshape(-1, 1) # Fake decreasing loss
            else:
                # Use 'numrows' last losses
                self.prevlosses = prevlosses[-(numrows):, :]
        elif self.train_settings.update_scheme == "weighted":
            numrows = self.train_settings.update_kwargs["weighted"]["save_last"]
            self.prevlosses = jnp.tile(init_loss, numrows).reshape((numrows, loss_shape))
        else:
            numrows = self.train_settings.update_kwargs["unweighted"]["save_last"]
            self.prevlosses = jnp.tile(init_loss, numrows).reshape((numrows, loss_shape))
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
                                 self.train_settings.jitted_update,
                                 verbose=True,
                                 verbose_kwargs={"print_every": self.logging.print_every})
        return
    
    def _initialized(self) -> bool:
        has_params = hasattr(self, "params")
        has_train_points = hasattr(self, "train_points")
        has_eval_points = hasattr(self, "eval_points")
        return has_params and has_train_points and has_eval_points

 
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
            s += "\n\n"

        return s
    

    # def _update_unweighted(self,
    #                        opt_state: optax.OptState,
    #                        params: optax.Params,
    #                        inputs: dict[str],
    #                        true_val: dict[str] | None = None,
    #                        update_key: int | None = None,
    #                        prevlosses: jax.Array | None = None):
    #     """
        
    #     """

    #     # Compute loss and gradients
    #     (total_loss, aux), grads = jax.value_and_grad(self._total_loss, has_aux=True)(
    #         params, inputs, true_val=true_val, update_key=update_key, prevlosses=prevlosses)

    #     # Apply updates
    #     updates, opt_state = self.optimizer.update(grads, opt_state, params)
    #     params = optax.apply_updates(params, updates)

    #     # Return updated params and state as well as the losses
    #     return params, opt_state, total_loss, *aux
    