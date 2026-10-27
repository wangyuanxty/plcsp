import copy
import json
import os
import time
import numpy as np
from plcsp_v4.dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action, machine_SPT_action, transbot_SPT_action
from plcsp_v4 import dfjspt_params
from plcsp_v4.dfjspt_data.radar_microwave_case_data import create_case_instance
from plcsp_v4.dfjspt_data.load_data_from_Ham import load_instance
from plcsp_v4.dfjspt_env_for_benchmark import DfjsptMaEnv


n_job1 = 1
n_job2 = 4
n_job3 = 4
n_transbots = 2
case_instance = create_case_instance(n_job1, n_job2, n_job3, n_transbots)

env = DfjsptMaEnv({"instance": case_instance})
makespan_list = []
for _ in range(1):
    observation, info = env.reset()
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions, real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            transbot_action = transbot_EET_action(real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    make_span = env.final_makespan
    makespan_list.append(make_span)
mean_makespan = np.mean(makespan_list)

print(makespan_list)
print(mean_makespan)







