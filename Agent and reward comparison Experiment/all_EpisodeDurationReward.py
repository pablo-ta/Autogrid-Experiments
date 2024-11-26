import sys
import os
from pathlib import Path

AutoGridPath = os.path.dirname(Path(__file__).parent.parent.absolute())
sys.path.append(AutoGridPath)

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from grid2op.Reward import EpisodeDurationReward
from base_IncreasingFlatReward import *

SAVE_PATH = "./agents_EpisodeDurationReward"


# Function to create the configuration for the main execution
def get_config():
    config["core"]["logger"]["file"]["filename"]= os.path.join(SAVE_PATH,"all_execution_log.log")
    config["common"]["save_path"] = SAVE_PATH
    config["common"]["load_path"] = SAVE_PATH
    config["common"]["env"]["env_kwargs"]["reward_class"] = EpisodeDurationReward
    config["common"]["agent"]["net_kwargs"]["tensorboard_log"] = os.path.join(SAVE_PATH, "tensorboard_log")

    return config


if __name__ == "__main__":
    main = AutoGrid.main(get_config(), force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()