import json
import os
import time
import pandas as pd
from ray.rllib import Policy

try:
    import gymnasium as gym
    gymnasium = True
except Exception:
    import gym
    gymnasium = False
import ray
from ray.rllib.models import ModelCatalog
import numpy as np
from plcsp_v4 import dfjspt_params
from plcsp_v4.dfjspt_data.load_data_from_Ham import load_instance
from plcsp_v4.dfjspt_env import DfjsptMaEnv
from plcsp_v4.dfjspt_agent_model import JobActionMaskModel, MachineActionMaskModel, TransbotActionMaskModel


ray.init(local_mode=False)

time0 = time.time()

ModelCatalog.register_custom_model(
    "job_agent_model", JobActionMaskModel
)
ModelCatalog.register_custom_model(
    "machine_agent_model", MachineActionMaskModel
)
ModelCatalog.register_custom_model(
    "transbot_agent_model", TransbotActionMaskModel
)

checkpoint_path = os.path.dirname(__file__) + "/training_results/J" + str(
            dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots) + '/MyTrainable_DfjsptMaEnv_bfb1a_00000_0_2025-03-03_17-49-07/checkpoint_000010'

job_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent0'
job_policy = Policy.from_checkpoint(job_policy_checkpoint_path)
# weights_j = job_policy.get_weights()
machine_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent1'
machine_policy = Policy.from_checkpoint(machine_policy_checkpoint_path)
transbot_policy_checkpoint_path = checkpoint_path + '/policies/policy_agent2'
transbot_policy = Policy.from_checkpoint(transbot_policy_checkpoint_path)

time1 = time.time()
print(f"Time for loading policies is {time1-time0}.")

num_repeat = 20
all_makespans = []
average_running_time = []
test_instances = dfjspt_params.n_instances_for_testing
average_makespans = np.zeros((test_instances,), dtype=float)
env = DfjsptMaEnv({
    "train_or_eval_or_test": "test",
})
for test_id in range(test_instances):
    print(f"Test instance {test_id + 1}: ")
    makespans = []
    time5 = time.time()
    for trial_id in range(num_repeat):
        # env = DfjsptMaEnv({"instance": test_instance})

        # observation, info = env.reset()
        observation, info = env.reset(options={
            "instance_id": test_id,
        })
        # env.render()
        done = False
        count = 0
        stage = next(iter(info["agent0"].values()), None)
        total_reward = 0

        while not done:
            if stage == 0:
                job_action = {
                    "agent0": job_policy.compute_single_action(obs=observation["agent0"], explore=True)[0]
                }
                observation, reward, terminated, truncated, info = env.step(job_action)
                stage = next(iter(info["agent1"].values()), None)

            elif stage == 1:
                machine_action = {
                    "agent1": machine_policy.compute_single_action(obs=observation["agent1"], explore=True)[0]
                }
                observation, reward, terminated, truncated, info = env.step(machine_action)
                stage = next(iter(info["agent2"].values()), None)

            else:
                transbot_action = {
                    "agent2": transbot_policy.compute_single_action(obs=observation["agent2"], explore=True)[0]
                }
                observation, reward, terminated, truncated, info = env.step(transbot_action)
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
                count += 1
                total_reward += reward["agent0"]

        make_span = env.final_makespan
        makespans.append(make_span)
    time6 = time.time()
    average_running_time.append((time6 - time5) / num_repeat)
    print(f"Average time for test instance {test_id + 1} is {(time6 - time5) / num_repeat}")
    mean_value = np.min(makespans)
    print(f"DRL average makespan for test instance {test_id + 1} is {mean_value}")
    average_makespans[test_id] = mean_value

print("Average makespans for all test instances:")
print(average_makespans)
print(f"Final Average makespan value is {np.mean(average_makespans)}")


print("Running time:")
print(average_running_time)
print(f"Average running time is {np.mean(average_running_time)}")


ray.shutdown()


