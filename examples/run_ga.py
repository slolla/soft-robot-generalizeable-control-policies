import random
import numpy as np

from ga.run import run_ga

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    
    run_ga(
        pop_size = 25,
        structure_shape = (5,5),
        experiment_name = "walker-no-normalization",
        max_evaluations = 200,
        train_iters = 1000,
        num_cores = 16,
        root_dir="/data/"
    )