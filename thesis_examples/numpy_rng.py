import numpy as np

def get_random_numbers(n):
    return [np.random.rand() for _ in range(n)]

np.random.seed(1234)
random_numbers = get_random_numbers(4)
[print(r) for r in random_numbers]
