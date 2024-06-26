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


class DeepONet(Model):
    """
    General DeepONet model:

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
        return

    @override
    def init_model(self, network_settings: list[dict]) -> None:
        
        # Number of networks in model
        num_branch_nets = len(network_settings["branch"])
        
        # Initialize network classes
        self.branch_nets = [setup_network(net) for net in network_settings["branch"]]
        self.trunk_net = setup_network(network_settings["trunk"])
                
        # Initialize network parameters
        self._key, *net_keys = jax.random.split(self._key, num_branch_nets+2)
        branch_params = [net.init(net_keys[i], jnp.ones((net.input_dim))) for i, net in enumerate(self.branch_nets)]
        trunk_params = self.trunk_net.init(net_keys[-1], jnp.ones((self.trunk_net.input_dim)))
        self.params = {"branch"+str(i): par for i, par in enumerate(branch_params)}
        self.params.update({"trunk": trunk_params})

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
                    scalars,
                    scalar_names: str | None = None,
                    tag: str | None = None,
                    step: int | None = None,
                    log: bool = False,
                    all_losses: jnp.ndarray | None = None):
        if log:
            writer = SummaryWriter(log_dir=self.dir.log_dir)
            writer.add_scalars(tag,
                            {name: np.array(loss) for name, loss in zip(scalar_names, scalars)},
                            global_step=step)
                    
            writer.close()
        
        return jnp.concatenate([all_losses, scalars.reshape(-1, scalars.shape[0])])
    
    @timer
    def plot_training_points(self, save=True, log=False, step=None):
        plt.figure()
        
        plt.scatter(self.eval_points_branch[0], self.eval_points_branch[1], color="orange", s=100)
        
        plt.scatter(self.train_points_branch[:, 0], self.train_points_branch[:, 1])
        
        if save:
            save_fig(self.dir.figure_dir, "training_points_branch", "png", plt.gcf())
        
        if log:
            log_figure(fig=plt.gcf(), name="training_points_branch", log_dir=self.dir.log_dir, step=step)
            
        plt.close()
        
        plt.figure()
        _ = jtu.tree_map_with_path(lambda x, y: plt.scatter(np.array(y)[:,0], np.array(y)[:,1], **self.sample_plots.kwargs[x[0].key]), OrderedDict(self.train_points_trunk[0]))
        
        if save:
            save_fig(self.dir.figure_dir, "training_points_trunk", "png", plt.gcf())
        
        if log:
            log_figure(fig=plt.gcf(), name="training_points_trunk", log_dir=self.dir.log_dir, step=step)
            
        plt.close()