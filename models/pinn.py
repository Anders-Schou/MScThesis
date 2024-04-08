from . import Model
from .optim import get_update
import optax


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
        if not hasattr(self, "params"):
            self.params = None
        self._parse_network_settings(settings["model"]["pinn"]["network"])
        self._parse_geometry_settings(settings["geometry"])

        # self._set_update()
        return

    def _parse_network_settings(self, network_settings):
        self.network_settings = network_settings
    
    def _parse_geometry_settings(self, geometry_settings):
        self.geometry_settings = geometry_settings

    def _set_loss(self):
        """
        Method for setting loss function.
        """
        pass

    def _set_update(self, update_fun_name: str, loss_fun_name: str = "total_loss"):
        """
        Method for setting update function.

        Currently supported:
            The usual unweighted update
            The SoftAdapt update
        
        """

        if not self.do_train:
            if self._verbose.training:
                print("Update method not set: Model is not set to train.")
            return
        
        # TODO: Remove fixed optimizer and utilize optimizer specified in settings 

        # Set optimizer and initial state
        self.schedule = optax.exponential_decay(self.train_settings.learning_rate,
                                                self.train_settings.decay_steps,
                                                self.train_settings.decay_rate)
        self.optimizer = self.train_settings.optimizer(learning_rate=self.schedule)
        # self.opt_state = self.optimizer.init(self.params)

        # Get loss function
        loss_fun = getattr(self, loss_fun_name)

        # Choose update function
        update_fun = get_update(loss_fun,
                                self.optimizer,
                                update_scheme=self.train_settings.update_scheme,
                                jit_compile=self.train_settings.jitted_update)
        
        # Set update function as method
        setattr(self, update_fun_name, update_fun)
        return
    
    def predict(self, *args, **kwargs):
        """
        A basic method for the forward pass without inputting
        parameters. For external use.
        """
        return self.forward(self.params, *args, **kwargs)
        