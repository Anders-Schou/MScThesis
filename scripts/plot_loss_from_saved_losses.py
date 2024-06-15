timer = parse_arguments = PINN01 = jnp = None

if __name__ == "__main__":
    raw_settings = timer(parse_arguments)()
    pinn = timer(PINN01)(raw_settings)

    with open(pinn.dir.log_dir.joinpath('all_losses.npy'), "rb") as f:
        loss_history = jnp.load(f)
    loss_names = ["Domain", "Lower boundary", "Right boundary", "Upper boundary", "Left boundary", "Circle"]
    all_losses = jnp.array([loss_history[:,0]] + [loss_history[:, i] + loss_history[:, i+1] for i in [1, 3, 5, 7, 9]]).transpose()
    pinn.plot_loss(all_losses, {f"{loss_name}": key for key, loss_name in enumerate(loss_names)}, fig_dir=pinn.dir.figure_dir, name="losses.png", epoch_step=pinn.logging.log_every)