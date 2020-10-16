import numpy as np
import torch as th
from utils.logging import get_logger
import random
from run import run
import config_util as cu


if __name__ == '__main__':
    logger = get_logger()
    algorithm = 'coma'
    minigame = '8m'

    config = cu.config_copy(cu.get_config(algorithm, minigame))

    random_Seed = random.randrange(0, 16546)

    np.random.seed(random_Seed)
    th.manual_seed(random_Seed)
    config['env_args']['seed'] = random_Seed

    run(config, logger)