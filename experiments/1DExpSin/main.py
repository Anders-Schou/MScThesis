from models.pinn1d import ExpSin1DPINN
from setup.parsers import parse_arguments

if __name__=="__main__":
    raw_settings = parse_arguments()
    pinn = ExpSin1DPINN(raw_settings)
    pinn.sample_points()
    pinn.train(update_key=5)
    pinn.plot_losses()
    pinn.plot_derivatives()
    