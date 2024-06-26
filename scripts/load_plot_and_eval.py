import sys
import argparse
import json


def load_json(path: str) -> dict:
    try:
        f = open(path, "r")
    except FileNotFoundError:
        print(f"Could not find settings file: '{path}'")
        exit()
    j = json.loads(f.read())
    f.close()
    return j


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=str, required=True)
    parser.add_argument("--mainfilepath", type=str, required=True)
    args = parser.parse_args()
    json_dict = load_json(args.settings)
    json_dict["io"]["settings_path"] = args.settings
    return json_dict, args.mainfilepath

if __name__ == "__main__":
    
    raw_settings, mainfilepath = parse_arguments()
    
    sys.path.insert(0, str(mainfilepath))

    from main import PINN01
    
    print(str(mainfilepath))
    
    pinn = PINN01(raw_settings)
    pinn.sample_eval_points()
    pinn.load_model()
    pinn.plot_results()
    
    f = open(pinn.dir.log_dir.joinpath('eval_error.dat'), "w")

    for metric_fun in ["mse", "maxabse", "L2rel"]:
        pinn.eval(metric=metric_fun)
        
        f.write(f'{metric_fun} xx error: {pinn.eval_result[metric_fun][0, 0]:.4f}\n')
        f.write(f'{metric_fun} xy error: {pinn.eval_result[metric_fun][0, 1]:.4f}\n')
        f.write(f'{metric_fun} yy error: {pinn.eval_result[metric_fun][1, 1]:.4f}\n')
        
    for metric_fun in ["mse", "maxabse", "L2rel"]:
        vm_error = pinn.eval(metric=metric_fun, cartesian=False)
            
        f.write(f'{metric_fun} vm_stress error: {vm_error:.4f}\n')