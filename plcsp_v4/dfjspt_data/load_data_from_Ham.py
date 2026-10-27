import os
import re
import numpy as np
from plcsp_v4 import dfjspt_params


def load_instance(instance_name: str):

    with open(instance_name, 'r') as file:

        n_jobs = int(file.readline().strip().split('=')[1].strip(';'))
        n_machines = int(file.readline().strip().split('=')[1].strip(';'))
        next(file)

        data = file.read()
        pattern = r"<(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)>"
        matches = re.findall(pattern, data)

        # processing_time_baseline = -1 * np.ones((n_jobs, n_machines))
        processing_time_matrix = -1 * np.ones((n_jobs, n_machines, n_machines))

        for match in matches:

            job_id = int(match[1]) - 1
            machine_id = int(match[2]) - 1
            operation_id = int(match[0]) - n_machines * job_id - 1
            processing_time = int(match[3])

            processing_time_matrix[job_id][operation_id][machine_id] = processing_time

    # for job in range(n_jobs):
    #     for operation in range(n_machines):
    #         processing_times = processing_time_matrix[job][operation]
    #         average_time = np.mean(processing_times[processing_times != -1])
    #         processing_time_baseline[job][operation] = average_time

    layout_data = [
        [0, 24, 25, 22, 33, 38, 35, 35, 35, 35, 34, 39, 27, 21, 40, 31, 38, 23, 30, 25],
        [24, 0, 24, 29, 37, 24, 27, 34, 27, 37, 30, 35, 37, 30, 20, 33, 28, 34, 25, 27],
        [25, 24, 0, 40, 32, 39, 35, 39, 36, 39, 24, 21, 38, 36, 39, 38, 40, 31, 32, 29],
        [22, 29, 40, 0, 21, 20, 20, 33, 31, 37, 39, 20, 35, 40, 20, 30, 33, 26, 20, 28],
        [33, 37, 32, 21, 0, 25, 38, 40, 33, 30, 29, 30, 27, 34, 31, 23, 27, 31, 35, 32],
        [38, 24, 39, 20, 25, 0, 35, 40, 26, 26, 40, 27, 37, 21, 21, 28, 40, 27, 20, 23],
        [35, 27, 35, 20, 38, 35, 0, 32, 37, 23, 28, 35, 37, 25, 33, 26, 23, 24, 27, 32],
        [35, 34, 39, 33, 40, 40, 32, 0, 37, 27, 37, 31, 25, 23, 34, 29, 34, 25, 20, 36],
        [35, 27, 36, 31, 33, 26, 37, 37, 0, 31, 32, 35, 35, 31, 38, 21, 39, 36, 28, 22],
        [35, 37, 39, 37, 30, 26, 23, 27, 31, 0, 22, 33, 39, 35, 36, 24, 25, 31, 23, 22],
        [34, 30, 24, 39, 29, 40, 28, 37, 32, 22, 0, 39, 33, 36, 29, 20, 32, 20, 31, 21],
        [39, 35, 21, 20, 30, 27, 35, 31, 35, 33, 39, 0, 40, 36, 29, 36, 28, 39, 23, 34],
        [27, 37, 38, 35, 27, 37, 37, 25, 35, 39, 33, 40, 0, 25, 26, 20, 35, 28, 36, 25],
        [21, 30, 36, 40, 34, 21, 25, 23, 31, 35, 36, 36, 25, 0, 28, 27, 40, 28, 25, 33],
        [40, 20, 39, 20, 31, 21, 33, 34, 38, 36, 29, 29, 26, 28, 0, 28, 27, 35, 27, 33],
        [31, 33, 38, 30, 23, 28, 26, 29, 21, 24, 20, 36, 20, 27, 28, 0, 31, 31, 29, 38],
        [38, 28, 40, 33, 27, 40, 23, 34, 39, 25, 32, 28, 35, 40, 27, 31, 0, 34, 40, 37],
        [23, 34, 31, 26, 31, 27, 24, 25, 36, 31, 20, 39, 28, 28, 35, 31, 34, 0, 40, 32],
        [30, 25, 32, 20, 35, 20, 27, 20, 28, 23, 31, 23, 36, 25, 27, 29, 40, 40, 0, 22],
        [25, 27, 29, 28, 32, 23, 32, 36, 22, 22, 21, 34, 25, 33, 33, 38, 37, 32, 22, 0]
    ]
    layout_data_array = np.array(layout_data)

    empty_moving_time_matrix = np.hstack((layout_data_array[:n_machines, :n_machines], np.zeros((n_machines, 1))))
    empty_moving_time_matrix = np.vstack((empty_moving_time_matrix, np.zeros((1, n_machines + 1))))
    transport_time_matrix = empty_moving_time_matrix

    n_operations_for_jobs = n_machines * np.ones((n_jobs,), dtype=int)
    job_arrival_time = np.zeros(shape=(n_jobs,), dtype=float)
    machine_quality = np.ones((1, n_machines), dtype=float)
    transbot_quality = np.ones((1, dfjspt_params.n_transbots), dtype=float)

    instance = [n_operations_for_jobs, job_arrival_time, processing_time_matrix, empty_moving_time_matrix,
                transport_time_matrix, machine_quality, transbot_quality]

    return instance

if __name__ == '__main__':
    instance_name = os.path.dirname(__file__) + "/Hurink_data/vdata/la01.dat"
    loaded_instance = load_instance(instance_name=instance_name)
    print(loaded_instance)

