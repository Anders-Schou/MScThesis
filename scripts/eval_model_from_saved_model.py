timer = parse_arguments = PINN01 = None

if __name__ == "__main__":
    
    raw_settings = timer(parse_arguments)()
    pinn = timer(PINN01)(raw_settings)
    timer(pinn.sample_points)()
    timer(pinn.load_model)()

    f = open(pinn.dir.log_dir.joinpath('time_and_eval.dat'), "w")

    for metric_fun in ["mse", "maxabse", "L2rel"]:
        timer(pinn.eval)(metric=metric_fun)
        
        f.write(f'L2-rel xx error: {pinn.eval_result[metric_fun][0, 0]:.4f}\n')
        f.write(f'L2-rel xy error: {pinn.eval_result[metric_fun][0, 1]:.4f}\n')
        f.write(f'L2-rel yy error: {pinn.eval_result[metric_fun][1, 1]:.4f}\n')
