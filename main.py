import numpy as np
import torch as th
from utils.logging import get_logger
import random
from run import standard_run
from offpg_run import offpg_run
import config_util as cu

'''
algorithm 설정 가이드(config/algs 경로의 파일이름 그대로)
만일 rnn agent의 QMIX 를 실행하고 싶다면 -> 'RNN_AGENT/qmix_beta'
만일 G2ANet agent COMA 를 실행하고 싶다면 -> 'G2ANet_Agent/coma'
만일 ROMA 를 실행하고 싶다면 -> 'Role_Learning_Agent/qmix_smac_latent'

mini game 설정 가이드
마린 3 vs 마린 3 -> '3m'
마린 8 vs 마린 3 -> '8m'
질럿2 추적자 3 vs 질럿2 추적자 3 -> '2s3z'
'''


if __name__ == '__main__':
    logger = get_logger()
    #algorithm = 'RNN_AGENT/qmix_beta'
    #algorithm = 'RNN_AGENT/coma'
    #algorithm = 'RNN_AGENT/offpg_smac'
    #algorithm = 'Role_Learning_Agent/rode'
   # algorithm = 'Role_Learning_Agent/rode_gat'
    #algorithm = 'Role_Learning_Agent/qmix_smac_latent'
    # algorithm = 'RNN_AGENT/noisemix_smac'
    # algorithm = 'RNN_AGENT/noisemix_smac'
    #algorithm = 'G2ANet_Agent/qmix_beta'
    algorithm = 'RNN_AGENT/graphmix'
    minigame = '2s3z'

    config = cu.config_copy(cu.get_config(algorithm, minigame))

    random_Seed = random.randrange(0, 16546)

    np.random.seed(random_Seed)
    th.manual_seed(random_Seed)
    config['env_args']['seed'] = random_Seed

    is_offpg = config['off_pg']

    if is_offpg:
        offpg_run(config, logger, minigame)
    else:
        standard_run(config, logger, minigame)