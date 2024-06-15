timer = parse_arguments = PINN01 = None

if __name__ == "__main__":
    raw_settings = timer(parse_arguments)()
    pinn = timer(PINN01)(raw_settings)
    timer(pinn.sample_points)()
    timer(pinn.load_model)()
    timer(pinn.plot_results)()