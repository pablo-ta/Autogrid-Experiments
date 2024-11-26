import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from grid2op.Reward import CombinedScaledReward, BridgeReward, CloseToOverflowReward, EconomicReward, DistanceReward, \
    EpisodeDurationReward, L2RPNReward, RedispReward
from experiments.sb3_grid_ace.final.base_IncreasingFlatReward import *

SAVE_PATH = "./agents_combined"

def create_combined_reward_env(**kwargs):
    env = grid2op.make(**kwargs)
    cr = env.get_reward_instance()
    cr.addReward("Bridge", BridgeReward(), 0.2)
    cr.addReward("CloseToOverflow", CloseToOverflowReward(), 0.2)
    cr.addReward("Distance", DistanceReward(), 0.2)
    cr.addReward("Economic", EconomicReward(), 0.2)
    cr.addReward("EpisodeDuration", EpisodeDurationReward(), 0.2)
    cr.addReward("RedispReward",RedispReward(), 0.2)
    cr.initialize(env)
    log.debug(F"Configured rewards: {env.get_reward_instance().rewards}")
    log.debug(F"Reward Range: [{cr._sum_min}, {cr._sum_max}], [{cr.reward_min}, {cr.reward_max}]")

    return env
# Function to create the configuration for the main execution
def get_config():

    config["core"]["logger"]["file"]["filename"]= os.path.join(SAVE_PATH,"all_execution_log.log")

    config["common"]["save_path"] = SAVE_PATH
    config["common"]["load_path"] = SAVE_PATH
    config["common"]["env"]["env_class"]=create_combined_reward_env
    config["common"]["env"]["env_kwargs"]["reward_class"]=CombinedScaledReward
    config["common"]["env"]["evaluation_env_class"]=grid2op.make

    #config["experiments"].pop("experiment_Do_Nothing")
    config["common"]["agent"]["net_kwargs"]["tensorboard_log"] = os.path.join(SAVE_PATH, "tensorboard_log")

    return config


if __name__ == "__main__":
    main = AutoGrid.main(get_config(), force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
