import sys
import argparse
import json

import jax.numpy as jnp


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

    with open(pinn.dir.log_dir.joinpath('all_losses.npy'), "rb") as f:
        loss_history = jnp.load(f)
    loss_names = ["Domain", "Lower boundary", "Right boundary", "Upper boundary", "Left boundary", "Circle"]
    all_losses = jnp.array([loss_history[:,0]] + [loss_history[:, i] + loss_history[:, i+1] for i in [1, 3, 5, 7, 9]]).transpose()
    pinn.plot_loss(all_losses, {f"{loss_name}": key for key, loss_name in enumerate(loss_names)}, fig_dir=pinn.dir.figure_dir, name="losses", epoch_step=pinn.logging.log_every)