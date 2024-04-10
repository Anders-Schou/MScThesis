from typing import override
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from torch.utils.tensorboard import SummaryWriter

from .base import Model
from .networks import setup_network
from utils.plotting import save_fig, log_figure
from utils.utils import timer


class PINN(Model):
    """
    General PINN model:

    MANDATORY methods to add:
    
        self.forward():
            A forward method, i.e. a forward pass through a network.
        
        self.loss_terms():
            A method that evaluates each loss term. These terms can
            be specified in separate methods if desired. This method
            should return a jax.Array of the loss terms.
        
        self.update():
             A method for updating parameters during training.
        
        self.train():
            A method for training the model. Typically calling the
            'update' method in a loop.
        
        self.eval():
            A method for evaluating the model.        

    
    OPTIONAL methods to add:

        self.total_loss():
            A function summing the loss terms. Can
            be rewritten to do something else.
        
        self.save_state():
            A method for saving the model state.
        
        self.load_state():
            A method for loading the model state.
    
    
    """

    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        
        self._parse_geometry_settings(settings["geometry"])
        self.init_model(settings["model"]["pinn"]["network"])
        return

    @override
    def init_model(self, network_settings: list[dict]) -> None:
        
        # Number of networks in model
        num_nets = len(network_settings)

        # Initialize network classes
        self.net = [setup_network(net) for net in network_settings]
        
        # Initialize network parameters
        self._key, *net_keys = jax.random.split(self._key, num_nets+1)
        params = [net.init(net_keys[i], jnp.ones((1, net.input_dim))) for i, net in enumerate(self.net)]
        self.params = {net.name+str(i): params[i] for i, net in enumerate(self.net)}

        # Set optimizer if relevant
        if self.do_train:
            self.schedule = optax.exponential_decay(self.train_settings.learning_rate,
                                                    self.train_settings.decay_steps,
                                                    self.train_settings.decay_rate)
            self.optimizer = self.train_settings.optimizer(learning_rate=self.schedule)
            # self.opt_state = self.optimizer.init(self.params)
        return
    
    def _parse_geometry_settings(self, geometry_settings):
        self.geometry_settings = geometry_settings
        return
    
    def predict(self, *args, **kwargs):
        """
        A basic method for the forward pass without inputting
        parameters. For external use.
        """
        return self.forward(self.params, *args, **kwargs)

    def log_scalars(self,
                params,
                inputs: dict[str, jax.Array],
                true_val: dict[str, jax.Array],
                update_key: int | None = None,
                step: int | None = None):
        losses = self._loss_terms(params, inputs, true_val, update_key)
        writer = SummaryWriter(log_dir=self.dir.log_dir)
        
        writer.add_scalars('Losses' , {self.loss_names[i]: np.array(losses[i]) for i in range(len(losses))}, global_step=step)
                
        writer.close()
        
        return
    
    @timer
    def plot_training_points(self, save=True, log=False, step=None):
        plt.figure()
        _ = jtu.tree_map_with_path(lambda x, y: plt.scatter(np.array(y)[:,0], np.array(y)[:,1], **self.plot_kwargs[x[0].key]), OrderedDict(self.train_points))
        
        if save:
            save_fig(self.dir.figure_dir, "training_points.png", "png", plt.gcf())
        
        if log:
            log_figure(fig=plt.gcf(), name="training_points", log_dir=self.dir.log_dir, step=step)
            
        plt.close()