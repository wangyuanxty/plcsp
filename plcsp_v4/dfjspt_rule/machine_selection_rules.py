import numpy as np


def machine_EET_action(legal_machine_actions, real_machine_attrs):
    machine_actions_mask = (1 - legal_machine_actions) * 1e8
    machine_last_finish_time = real_machine_attrs[:, 3]
    # machine_last_finish_time = np.zeros(len(legal_machine_actions))
    # machine_last_finish_time[:len(real_machine_attrs)] = [obs[3] for obs in real_machine_attrs]
    machine_last_finish_time += machine_actions_mask
    # EET_machine_action = {
    #     "agent1": np.argmin(machine_last_finish_time)
    # }
    min_index = np.argmin(machine_last_finish_time)
    min_indices = np.where(machine_last_finish_time == machine_last_finish_time[min_index])[0]
    EET_machine_action = {
        "agent1": np.random.choice(min_indices)
    }
    return EET_machine_action


def machine_SPT_action(legal_machine_actions, real_machine_attrs):
    machine_actions_mask = (1 - legal_machine_actions) * 1e8
    machine_processing_time = real_machine_attrs[:, 5]
    # machine_processing_time = np.zeros(len(legal_machine_actions))
    # machine_processing_time[:len(real_machine_attrs)] = [obs[5] for obs in real_machine_attrs]
    machine_processing_time += machine_actions_mask
    SPT_machine_action = {
        "agent1": np.argmin(machine_processing_time)
    }
    return SPT_machine_action


def transbot_EET_action(legal_transbot_actions, real_transbot_attrs):
    transbot_actions_mask = (1 - legal_transbot_actions) * 1e8
    transbot_last_finish_time = real_transbot_attrs[:, 3]
    # transbot_last_finish_time = np.zeros(len(legal_transbot_actions))
    # transbot_last_finish_time[:len(real_transbot_attrs)] = [obs[3] for obs in real_transbot_attrs]
    transbot_last_finish_time += transbot_actions_mask
    # EET_transbot_action = {
    #     "agent2": np.argmin(transbot_last_finish_time)
    # }
    min_index = np.argmin(transbot_last_finish_time)
    min_indices = np.where(transbot_last_finish_time == transbot_last_finish_time[min_index])[0]
    EET_transbot_action = {
        "agent2": np.random.choice(min_indices)
    }
    return EET_transbot_action


def transbot_SPT_action(legal_transbot_actions, real_transbot_attrs):
    transbot_actions_mask = (1 - legal_transbot_actions) * 1e8
    transbot_transporting_time = real_transbot_attrs[:, 6]
    # transbot_transporting_time = np.zeros(len(legal_transbot_actions))
    # transbot_transporting_time[:len(real_transbot_attrs)] = [obs[6] for obs in real_transbot_attrs]
    transbot_transporting_time += transbot_actions_mask
    SPT_transbot_action = {
        "agent2": np.argmin(transbot_transporting_time)
    }
    return SPT_transbot_action