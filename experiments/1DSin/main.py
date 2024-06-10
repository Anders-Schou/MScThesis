from time import perf_counter

from models.pinn1d import Sin1DPINN
from setup.parsers import parse_arguments

if __name__=="__main__":
    t1 = perf_counter()
    
    raw_settings = parse_arguments()
    pinn = Sin1DPINN(raw_settings)
    pinn.sample_points()
    pinn.train(update_key=None)
    pinn.write_model()
    pinn.plot_losses()
    pinn.plot_derivatives()
    
    t2 = perf_counter()
    
    f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w+")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')