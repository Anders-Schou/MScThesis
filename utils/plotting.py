import os

import matplotlib.pyplot as plt

def save_fig(dir: str, file_name: str, format: str = "png"):
    if not file_name.endswith("." + format):
        file_name += ("." + format)
    plt.savefig(os.path.join(dir, file_name), format=format)
