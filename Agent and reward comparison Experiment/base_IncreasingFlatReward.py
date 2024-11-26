import sys
import os
from pathlib import Path


AutoGridPath = os.path.dirname(Path(__file__).parent.parent.parent.absolute())
sys.path.append(AutoGridPath)

from src.helpers import create_experiment_gitignore

from grid2op.Agent import BaseAgent, DoNothingAgent
from grid2op.Reward import L2RPNWCCI2022ScoreFun, IncreasingFlatReward, L2RPNSandBoxScore

from src.agents.Grid2OpSB3 import SB3AgentGrid2Op
import grid2op
from src import constants, AutoGrid
from stable_baselines3 import DDPG, SAC, TD3, DQN, PPO, A2C
from stable_baselines3.a2c import MlpPolicy as a2cMlpPolicy
from stable_baselines3.ddpg import MlpPolicy as ddpgMlpPolicy
from stable_baselines3.sac import MlpPolicy as sacMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3MlpPolicy
from stable_baselines3.dqn import MlpPolicy as dqnMlpPolicy
from stable_baselines3.ppo import MlpPolicy as ppoMlpPolicy
from src.makers.SB3 import create_agent_sb3
from src.envs.gymenv_heuristics import GymEnvWithHeuristics
from grid2op.gym_compat import GymEnv, BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace
from typing import List
from grid2op.Action import BaseAction
import numpy as np
from lightsim2grid import LightSimBackend
import logging

from grid2op.Converter import IdToAct

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

def _filter_action(action):
    MAX_ELEM = 1
    act_dict = action.impact_on_objects()
    #forbidden actions:
    forbidden_elem = 0
    forbidden_elem += act_dict["force_line"]["reconnections"]["count"]
    forbidden_elem += act_dict["force_line"]["disconnections"]["count"]
    forbidden_elem += act_dict["switch_line"]["count"]
    forbidden_elem += len(act_dict["topology"]["bus_switch"])
    forbidden_elem += len(act_dict["topology"]["assigned_bus"])
    forbidden_elem += len(act_dict["topology"]["disconnect_bus"])
    if forbidden_elem > 0:
        return False
    elem = 0
    elem += len(act_dict["redispatch"]["generators"])
    elem += len(act_dict["curtailment"]["limit"])
    elem += len(act_dict["storage"]["capacities"])
    if elem <= MAX_ELEM:
        return True
    return False
"""
04-19 18:38 | DEBUG   | __main__                 | 119  | Original Action space size:134283
04-19 18:38 | DEBUG   | __main__                 | 121  | Filtered Action space size:201
"""
def create_discrete_action_space(env_action_space, load_path=None, save_path=None, agent_name=None,
                                 _action_filter=None):
    converter = IdToAct(env_action_space)
    if load_path is not None:
        try:
            my_path = load_path
            if agent_name:
                my_path = os.path.join(my_path, agent_name)
            converter.init_converter(all_actions=os.path.join(my_path, "filtered_actions.npy"))
            log.debug(F"Loaded Action space size:{converter.n} from folder [{my_path}]")
            return DiscreteActSpace(converter, action_list=converter.all_actions)
        except FileNotFoundError as e:
            log.warning(F"Could not load filtered action space, one will be created now. Error: {str(e)}")
            pass

    if (_action_filter is not None):
        converter.init_converter()
        log.debug(F"Original Action space size:{converter.n}")
        converter.filter_action(_action_filter)
        log.debug(F"Filtered Action space size:{converter.n}")
        if save_path is not None:
            log.debug(F"Saving filtered actions on {save_path}")
            my_path = load_path
            if agent_name:
                my_path = os.path.join(my_path, agent_name)
            if not os.path.exists(my_path):
                os.makedirs(my_path)
            converter.save(my_path, "filtered_actions")
    return DiscreteActSpace(converter, action_list=converter.all_actions)


class CustomGymEnv(GymEnvWithHeuristics):

    def __init__(self, env_init, *args, reward_cumul="init", safe_max_rho=0.9, **kwargs):
        super().__init__(env_init, reward_cumul=reward_cumul, *args, **kwargs)
        self._safe_max_rho = safe_max_rho
        self.dn = self.init_env.action_space({})

    def heuristic_actions(self, g2op_obs, reward, done, info) -> List[BaseAction]:
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)

        # default:_ play do nothing if there is "no problem" according to the "rule of thumb"
        res = []

        if np.any(to_reco):
            # reconnect something if it can be
            reco_id = np.where(to_reco)[0]
            for line_id in reco_id:
                g2op_act = self.init_env.action_space({"set_line_status": [(line_id, +1)]})
                res.append(g2op_act)
        return res





SAVE_PATH = "./agents_IncreasingFlatReward"

BoxActionSpace = {
    "class": BoxGymActSpace,
    "action_space_kwargs": {
        "attr_to_keep": [
            #"set_line_status",
            #"change_line_status",
            #"set_bus",
            #"change_bus",
            "redispatch",
            # "set_storage",
            "curtail"
        ]
    }
}

DiscreteSingleActionSpace = {
    "class": create_discrete_action_space,
    "action_space_kwargs": {
        "save_path": "./discrete_action_space",
        "load_path": "./discrete_action_space",
        "_action_filter": _filter_action,
    }
}

DiscreteMultipleActionSpace = {
    "class": DiscreteActSpace,
    "action_space_kwargs": {
        "attr_to_keep": {
            #"set_line_status",
            #"change_line_status",
            #"set_bus",
            #"change_bus",
            "redispatch",
            # "set_storage",
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
            "gymenv_class": CustomGymEnv
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
                # "policy_kwargs": {
                #    "net_arch": [256, 256, 128, 128, 64],
                # }
            }
        },
        "training": constants.TRAINING_DEFAULT,
        "training_kwargs": {
            "total_timesteps": 10_000_000,
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
        "experiment_A2C_box": {
            "name": "A2C_box",
            "action_space": BoxActionSpace,
            "agent": {
                "class": A2C,
                "net_kwargs": {
                    "policy": a2cMlpPolicy
                }
            },
        },
        "experiment_A2C_discrete": {
            "name": "A2C_discrete",
            "action_space": DiscreteMultipleActionSpace,
            "agent": {
                "class": A2C,
                "net_kwargs": {
                    "policy": a2cMlpPolicy
                }
            },
        },
        "experiment_A2C_discrete_singleaction": {
            "name": "A2C_discrete_singleaction",
            "action_space": DiscreteSingleActionSpace,
            "agent": {
                "class": A2C,
                "net_kwargs": {
                    "policy": a2cMlpPolicy
                }
            },
        },
         "experiment_DDPG_box": {  # CANT USE DISCRETE ACTION SPACE
             "name": "DDPG_box",
             "action_space": BoxActionSpace,
             "agent": {
                 "class": DDPG,
                 "net_kwargs": {
                     "policy": ddpgMlpPolicy
                 }
             },
         },
        # "experiment_SAC_box": {  # CANT USE DISCRETE ACTION SPACE
        #     "name": "SAC_box",
        #     "action_space": BoxActionSpace,
        #     "agent": {
        #         "class": SAC,
        #         "net_kwargs": {
        #             "policy": sacMlpPolicy
        #         }
        #     },
        # },
        # "experiment_TD3_box": {  # CANT USE DISCRETE ACTION SPACE
        #     "name": "TD3_box",
        #     "action_space": BoxActionSpace,
        #     "agent": {
        #         "class": TD3,
        #         "net_kwargs": {
        #             "policy": td3MlpPolicy
        #         }
        #     },
        # },
        "experiment_DQN_discrete": {
            "name": "DQN_discrete",
            "action_space": DiscreteMultipleActionSpace,
            "agent": {
                "class": DQN,
                "net_kwargs": {
                    "policy": dqnMlpPolicy
                }
            },
        },
        "experiment_DQN_discrete_singleaction": {
            "name": "DQN_discrete_singleaction",
            "action_space": DiscreteSingleActionSpace,
            "agent": {
                "class": DQN,
                "net_kwargs": {
                    "policy": dqnMlpPolicy
                }
            },
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
        "experiment_PPO_singleaction": {
            "name": "PPO_discrete_singleaction",
            "action_space": DiscreteSingleActionSpace,
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
