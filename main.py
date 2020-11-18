import numpy as np
import torch as th
from utils.logging import get_logger
import random
from run import run
import config_util as cu

'''
algorithm 설정 가이드(config/algs 경로의 파일이름 그대로)
만일 QMIX 를 실행하고 싶다면 -> 'qmix_beta'
만일 COMA 를 실행하고 싶다면 -> 'coma'

minigame 설정 가이드
마린 3 vs 마린 3 -> '3m'
마린 8 vs 마린 3 -> '8m'
질럿2 추적자 3 vs 질럿2 추적자 3 -> '2s3z'
'''

if __name__ == '__main__':
    logger = get_logger()
    algorithm = 'rode'
    minigame = '2s3z'

    config = cu.config_copy(cu.get_config(algorithm, minigame))

    random_Seed = random.randrange(0, 16546)

    np.random.seed(random_Seed)
    th.manual_seed(random_Seed)
    config['env_args']['seed'] = random_Seed

    run(config, logger)