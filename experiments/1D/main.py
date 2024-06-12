from time import perf_counter

from models.pinn1d import PINN1D
from setup.parsers import parse_arguments

if __name__=="__main__":
    t1 = perf_counter()
    
    raw_settings = parse_arguments()
    pinn = PINN1D(raw_settings, type="sin")
    pinn.sample_points()
    pinn.train(update_key=raw_settings["update_key"])
    pinn.write_model()
    pinn.plot_results(update_key=raw_settings["update_key"])
    
    t2 = perf_counter()
    
    f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    
    
    t1 = perf_counter()
    
    raw_settings = parse_arguments()
    raw_settings["io"].update({key: value for key, value in raw_settings["io2"].items()})
    pinn = PINN1D(raw_settings)
    pinn.sample_points()
    pinn.train(update_key=raw_settings["update_key"])
    pinn.write_model()
    pinn.plot_results(update_key=raw_settings["update_key"])
    
    t2 = perf_counter()
    
    f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')