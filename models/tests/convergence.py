
import jax
import jax.numpy


from models.pinn import PINN
from setup.parsers import parse_arguments

class Conv1DPINN(PINN):
    def __init__(self, settings: dict, *args, **kwargs):
        super().__init__(settings, *args, **kwargs)
        self.init_model(settings["model"]["pinn"]["network"])
        self._set_update(loss_fun_name="_total_loss", optimizer_name="optimizer")
        self.network = self.net[0]

    def forward(self, params, input: jax.Array) -> jax.Array:
        return self.network.apply(params["net0"], input)
    
if __name__ == "__main__":
    raw_settings = parse_arguments()
    pinn = Conv1DPINN(raw_settings)

