import sys
import os
from pathlib import Path


AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

from src.helpers import create_experiment_gitignore

from grid2op.Agent import DoNothingAgent
from grid2op.Reward import IncreasingFlatReward, L2RPNSandBoxScore

from src.agents.Grid2OpSB3 import SB3AgentGrid2Op
import grid2op
from src import constants, AutoGrid
from stable_baselines3 import  PPO
from stable_baselines3.ppo import MlpPolicy as ppoMlpPolicy
from src.makers.SB3 import create_agent_sb3
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace
from lightsim2grid import LightSimBackend
import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)



class Do_nothing_agent(SB3AgentGrid2Op):
    def __init__(self,
                 g2op_action_space,
                 gym_act_space,
                 gym_obs_space,
                 nn_type,
                 nn_path=None,
                 nn_kwargs=None,
                 custom_load_dict=None,
                 gymenv=None,
                 eval_env=None,
                 iter_num=None,
                 ):
        super().__init__(g2op_action_space, gym_act_space, gym_obs_space, nn_type, nn_path=nn_path, nn_kwargs=nn_kwargs,
                         custom_load_dict=custom_load_dict,
                         gymenv=gymenv, eval_env=eval_env, iter_num=iter_num)
        self.do_nothing_action = g2op_action_space({})
        # self.do_nothing_action_gym = gym_act_space({})

    def get_act(self, observation, reward, done):
        return 0

    def build(self):
        pass

    def learn(self,
              total_timesteps=1,
              save_path=None,
              callbacks={},
              **learn_kwargs):
        pass

def create_do_nothing_agent(experiment_config):
    Grid2OpAgentClass = experiment_config.get("agent").get("agent", False)
    if Grid2OpAgentClass is False:
        Grid2OpAgentClass = DoNothingAgent

    env = experiment_config.get("env").get("env")
    return Grid2OpAgentClass(env.action_space)

SAVE_PATH = "./agents_IncreasingFlatReward"

BoxActionSpace = {
    "class": BoxGymActSpace,
    "action_space_kwargs": {
        "attr_to_keep": [
            "set_line_status",
            "change_line_status",
            "set_bus",
            "change_bus",
            "redispatch",
            "set_storage",
            "curtail"
        ]
    }
}


DiscreteMultipleActionSpace = {
    "class": DiscreteActSpace,
    "action_space_kwargs": {
        "attr_to_keep": {
            "set_line_status",
            "change_line_status",
            "set_bus",
            "change_bus",
            "redispatch",
            "set_storage",
            "curtail"
        }
    }
}

config = {
    "core": {
        "logger": {
            "console": {
                "level": "INFO"
            },
            "file": {
                "level": "DEBUG",
                "filename": os.path.join(SAVE_PATH,"all_execution_log.log"),
                "mode": "a"
            }
        }
    },
    "common": {
        "save_path": SAVE_PATH,
        "load_path": SAVE_PATH,
        "env": {
            "simulator": grid2op,
            "env_class": grid2op.make,
            "env_kwargs": {
                "dataset": "l2rpn_icaps_2021_large_train",
                "difficulty": "competition",
                "reward_class":IncreasingFlatReward,
                "backend": LightSimBackend
            },
            "evaluation_env_kwargs":{
                "dataset": "l2rpn_icaps_2021_large_val",
                "difficulty": "competition",
                "reward_class":L2RPNSandBoxScore,
                "backend": LightSimBackend
            },
            "gymenv_class": GymEnv
        },
        # "action_space": BoxActionSpace, SET AT EXPERIMENT LEVEL
        "observation_space": {
            "class": BoxGymObsSpace,
            "observation_space_kwargs": {
                "attr_to_keep": [
                    "gen_p",
                    # "gen_p_before_curtail",
                    "gen_q",
                    "gen_v",
                    "gen_margin_up",
                    "gen_margin_down",
                    "load_p",
                    "load_q",
                    "load_v",
                    "p_or",
                    "q_or",
                    "v_or",
                    "a_or",
                    "p_ex",
                    "q_ex",
                    "v_ex",
                    "a_ex",
                    "rho",
                    "line_status",
                    "timestep_overflow",
                    # "topo_vect",
                    # "time_before_cooldown_line",
                    # "time_before_cooldown_sub",
                    # "time_next_maintenance",
                    # "duration_next_maintenance",
                    "target_dispatch",
                    "actual_dispatch",
                    # "storage_charge",
                    # "storage_power_target",
                    # "storage_power",
                    "curtailment",
                    "curtailment_limit",
                    # "curtailment_limit_effective",
                    "thermal_limit",
                    "theta_or",
                    "theta_ex",
                    "load_theta",
                    "gen_theta"]
            }
        },
        "agent": {
            "maker": create_agent_sb3,
            "net_kwargs": {
                "tensorboard_log": os.path.join(SAVE_PATH, "tensorboard_log"),
            }
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "total_timesteps": 2_000_000,
            "reset_num_timesteps":False,
            "add_training":False,
        },
        "evaluation": constants.EVALUATION_GRID2OP,
        "evaluation_kwargs": {
            "nb_episode": 150,
        },
    },
    "experiments": {
        "experiment_Do_Nothing": {
            "name": "Do_Nothing",
            # "action_space": DiscreteMultipleActionSpace,
            "agent": {
                "maker": create_do_nothing_agent,
                "agent": DoNothingAgent,
            },
            "training": None,
        },
        "experiment_PPO_box": {
            "name": "PPO_box",
            "action_space": BoxActionSpace,
            "agent": {
                "class": PPO,
                "net_kwargs": {
                    "policy": ppoMlpPolicy
                }
            },
        },
        "experiment_PPO_discrete": {
            "name": "PPO_discrete",
            "action_space": DiscreteMultipleActionSpace,
            "agent": {
                "class": PPO,
                "net_kwargs": {
                    "policy": ppoMlpPolicy
                }
            },
        },
    }
}


# Function to create the configuration for the main execution
# Its mandatory to have this function declared with this signature.
def get_config():
    return config


if __name__ == "__main__":
    main = AutoGrid.main(config, force_log="DEBUG")
    create_experiment_gitignore(SAVE_PATH)
    log.info(F"Executing experiment file {__file__}")
    main.run()
