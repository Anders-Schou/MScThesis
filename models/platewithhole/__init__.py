"""
This module contains relevant code for the specific problem of
determining the stress throughout a plate with a hole in it
subject to boundary stress.
"""

from .analytic import polar_sol_true, cart_sol_true
from .loss import loss_rect, loss_circ_rr_rt, loss_dirichlet
from .pinn import PWHPINN
from .plotting import plot_potential, plot_loss
