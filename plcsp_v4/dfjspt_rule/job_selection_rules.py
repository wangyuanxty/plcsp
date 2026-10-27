import numpy as np


def job_EST_action(legal_job_actions, real_job_attrs):
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_last_finish_time = real_job_attrs[:, 2]
    # jobs_last_finish_time =  np.array([obs[2] for obs in real_job_attrs])
    jobs_last_finish_time += job_actions_mask
    EST_job_action = {
        "agent0": np.argmin(jobs_last_finish_time)
    }
    return EST_job_action

def job_FIFO_action(legal_job_actions, real_job_attrs):
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_last_finish_time = real_job_attrs[:, 2]
    # jobs_last_finish_time =  np.array([obs[2] for obs in real_job_attrs])
    jobs_last_finish_time += job_actions_mask
    min_index = np.argmin(jobs_last_finish_time)
    min_indices = np.where(jobs_last_finish_time == jobs_last_finish_time[min_index])[0]
    FIFO_job_action = {
        "agent0": np.random.choice(min_indices)
    }
    return FIFO_job_action


# def job_MOPNR_action(legal_job_actions, real_job_attrs):
#     job_actions_mask = (1 - legal_job_actions) * 1e8
#     jobs_remain_operations = env.n_operations_for_jobs -  np.array([obs[1] for obs in real_job_attrs])
#     jobs_remain_operations -= job_actions_mask
#     MOPNR_job_action = {
#         "agent0": np.argmax(jobs_remain_operations)
#     }
#     return MOPNR_job_action


def job_SPT_action(legal_job_actions, real_job_attrs):
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_process_time = real_job_attrs[:, 6]
    # jobs_process_time =  np.array([obs[6] for obs in real_job_attrs])
    jobs_process_time += job_actions_mask
    SPT_job_action = {
        "agent0": np.argmin(jobs_process_time)
    }
    return SPT_job_action


def job_MTWR_action(legal_job_actions, real_job_attrs):
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_remain_work = real_job_attrs[:, 7]
    # jobs_remain_work =  np.array([obs[7] for obs in real_job_attrs])
    jobs_remain_work -= job_actions_mask
    MTWR_job_action = {
        "agent0": np.argmax(jobs_remain_work)
    }
    return MTWR_job_action


def job_FDD_MTWR_action(legal_job_actions, real_job_attrs):
    job_actions_mask = (1 - legal_job_actions) * 1e8
    jobs_cumulative_finished_work = real_job_attrs[:, 5]
    # jobs_cumulative_finished_work = np.zeros(len(legal_job_actions))
    # jobs_cumulative_finished_work[:len(real_job_attrs)] = [obs[5] for obs in real_job_attrs]
    jobs_remain_work = real_job_attrs[:, 7]
    # jobs_remain_work = np.zeros(len(legal_job_actions))
    # jobs_remain_work[:len(real_job_attrs)] = [obs[7] for obs in real_job_attrs]
    if np.any(jobs_remain_work == 0):
        jobs_remain_work[jobs_remain_work == 0] = 0.001
    jobs_ratio = 1.0 * jobs_cumulative_finished_work / jobs_remain_work
    jobs_ratio += job_actions_mask
    FDD_MTWR_job_action = {
        "agent0": np.argmin(jobs_ratio)
    }
    return FDD_MTWR_job_action




