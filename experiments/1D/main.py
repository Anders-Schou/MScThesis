from time import perf_counter

from models.pinn1d import PINN1D
from setup.parsers import parse_arguments

if __name__=="__main__":
    descriptions = ["Data-fitting", "1st order PINN", "2nd order PINN", "3rd order PINN", "4th order PINN", "4th order PINN where we seek to approximate the hessian"]
    ids = ["Data", "PINN1", "PINN2", "PINN3", "PINN4", "PINN5"]
    ids = [x + "/10000" for x in ids]
    for update_key, id in enumerate(ids):
        t1 = perf_counter()
        
        raw_settings = parse_arguments()
        raw_settings["id"] = id
        raw_settings["description"] = descriptions[update_key]
        pinn = PINN1D(raw_settings, type="sin")
        pinn.sample_points()
        epoch = pinn.train(update_key=update_key, early_stop_tol = raw_settings["run"]["train"]["early_stop_tol"])
        pinn.write_model()
        pinn.plot_results(update_key=update_key)
        pinn.eval()
        
        t2 = perf_counter()
        
        f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
        f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
        if epoch != -1:
            f.write(f'Converged to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {epoch} epochs\n')
        else:
            f.write(f'Did not converge to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {raw_settings["run"]["train"]["iterations"]} epochs\n')
        for fun_name in ["mse", "maxabse", "L2rel"]:
            for i in range(5):
                f.write(f'{fun_name} error {i}: {pinn.eval_result[fun_name][str(i)]:.8f}\n')
        
        
        t1 = perf_counter()
        
        raw_settings = parse_arguments()
        raw_settings["io"].update({key: value for key, value in raw_settings["io2"].items()})
        raw_settings["id"] = id
        raw_settings["description"] = descriptions[update_key]
        pinn = PINN1D(raw_settings, type="expsin")
        pinn.sample_points()
        epoch = pinn.train(update_key=update_key, early_stop_tol = raw_settings["run"]["train"]["early_stop_tol"])
        pinn.write_model()
        pinn.plot_results(update_key=update_key)
        pinn.eval()
        
        t2 = perf_counter()
        
        f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
        f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
        if epoch != -1:
            f.write(f'Converged to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {epoch} epochs\n')
        else:
            f.write(f'Did not converge to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {raw_settings["run"]["train"]["iterations"]} epochs\n')
        for fun_name in ["mse", "maxabse", "L2rel"]:
            for i in range(5):
                f.write(f'{fun_name} error {i}: {pinn.eval_result[fun_name][str(i)]:.8f}\n')
    
    
    # t1 = perf_counter()

    # raw_settings = parse_arguments()
    # pinn = PINN1D(raw_settings, type="sin")
    # pinn.sample_points()
    # epoch = pinn.train(update_key=raw_settings["update_key"], early_stop_tol = raw_settings["run"]["train"]["early_stop_tol"])
    # pinn.write_model()
    # pinn.plot_results(update_key=raw_settings["update_key"])
    # pinn.eval()

    # t2 = perf_counter()

    # f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    # f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    # if epoch != -1:
    #     f.write(f'Converged to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {epoch} epochs\n')
    # else:
    #     f.write(f'Did not converge to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {raw_settings["run"]["train"]["iterations"]} epochs\n')
    # for fun_name in ["mse", "maxabse", "L2rel"]:
    #     for i in range(5):
    #         f.write(f'{fun_name} error {i}: {pinn.eval_result[fun_name][str(i)]:.8f}\n')
    
    
    # t1 = perf_counter()

    # raw_settings = parse_arguments()
    # raw_settings["io"].update({key: value for key, value in raw_settings["io2"].items()})
    # pinn = PINN1D(raw_settings, type="expsin")
    # pinn.sample_points()
    # epoch = pinn.train(update_key=raw_settings["update_key"], early_stop_tol = raw_settings["run"]["train"]["early_stop_tol"])
    # pinn.write_model()
    # pinn.plot_results(update_key=raw_settings["update_key"])
    # pinn.eval()

    # t2 = perf_counter()

    # f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")
    # f.write(f'Time taken for the whole training process: {t2-t1:.1f} s \t or \t {(t2-t1)/60.0:.1f} min\n')
    # if epoch != -1:
    #     f.write(f'Converged to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {epoch} epochs\n')
    # else:
    #     f.write(f'Did not converge to a tol of {raw_settings["run"]["train"]["early_stop_tol"]} in {raw_settings["run"]["train"]["iterations"]} epochs\n')
    # for fun_name in ["mse", "maxabse", "L2rel"]:
    #     for i in range(5):
    #         f.write(f'{fun_name} error {i}: {pinn.eval_result[fun_name][str(i)]:.8f}\n')