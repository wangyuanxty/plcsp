import copy
# import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from plcsp_v4.dfjspt_env_for_imitation import DfjsptMaEnv
from plcsp_v4.dfjspt_rule.job_selection_rules import job_FDD_MTWR_action
from plcsp_v4.dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action
from ray.rllib.models.preprocessors import get_preprocessor


def generate_sample_batch(batch_type):
    job_batch_builder = SampleBatchBuilder()
    machine_batch_builder = SampleBatchBuilder()
    transbot_batch_builder = SampleBatchBuilder()

    env = DfjsptMaEnv()
    job_prep = get_preprocessor(env.observation_space["agent0"])(env.observation_space["agent0"])
    machine_prep = get_preprocessor(env.observation_space["agent1"])(env.observation_space["agent1"])
    transbot_prep = get_preprocessor(env.observation_space["agent2"])(env.observation_space["agent2"])

    observation, info = env.reset()
    t = 0
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            job_prev_obs = copy.deepcopy(observation["agent0"])
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            FDD_MTWR_job_action = job_FDD_MTWR_action(legal_job_actions=legal_job_actions,
                                                      real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(FDD_MTWR_job_action)
            stage = next(iter(info["agent1"].values()), None)
            if batch_type == "job":
                job_batch_builder.add_values(
                    obs_flat=job_prep.transform(job_prev_obs),
                    # obs=job_prev_obs,
                    actions=FDD_MTWR_job_action["agent0"],
                )
            t += 1

        elif stage == 1:
            machine_prev_obs = copy.deepcopy(observation["agent1"])
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)
            if batch_type == "machine":
                machine_batch_builder.add_values(
                    obs_flat=machine_prep.transform(machine_prev_obs),
                    # obs=machine_prev_obs,
                    actions=EET_machine_action["agent1"],
                )
            t += 1
        else:
            transbot_prev_obs = copy.deepcopy(observation["agent2"])
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            EET_transbot_action = transbot_EET_action(legal_transbot_actions=legal_transbot_actions,
                                                      real_transbot_attrs=real_transbot_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            if batch_type == "transbot":
                transbot_batch_builder.add_values(
                    obs_flat=transbot_prep.transform(transbot_prev_obs),
                    # obs=transbot_prev_obs,
                    actions=EET_transbot_action["agent2"],
                )
            t += 1

            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    if batch_type == "job":
        return job_batch_builder.build_and_reset()
    elif batch_type == "machine":
        return machine_batch_builder.build_and_reset()
    elif batch_type == "transbot":
        return transbot_batch_builder.build_and_reset()
    else:
        raise RuntimeError(f"Invalid batch type: {batch_type}!")
# my_job_batch = generate_sample_batch("job")
# my_machine_batch = generate_sample_batch("machine")
# my_transbot_batch = generate_sample_batch("transbot")
