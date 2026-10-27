import os
import numpy as np
from typing import List
from plcsp_v4 import dfjspt_params
import json

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

def generate_instance_pool(
        seed,
        n_instances,
        max_n_jobs,
        max_n_machines,
        max_n_transbots,
) -> List:
    instance_pool = []

    # np.random.seed(seed)

    for i in range(n_instances):
        np.random.seed(seed + i)

        # n_jobs = np.random.randint(10, max_n_jobs + 1)
        n_jobs = 15
        n_machines = 15
        # n_jobs_for_this_instance = max_n_jobs
        # n_machines = np.random.randint(5, max_n_machines + 1)
        # n_machines = 5 * np.random.randint(1, 4)

        # n_transbots = np.random.randint(10, max_n_transbots + 1)
        n_transbots = 2

        # n_operations_for_jobs = np.random.randint(int(n_machines * 0.8), int(n_machines * 1.2) + 1, size=n_jobs,
        #                                           dtype=int)
        n_operations_for_jobs = n_machines * np.ones((n_jobs,), dtype=int)

        job_arrival_time = np.zeros(shape=(n_jobs,), dtype=int)

        processing_time_baseline = np.random.randint(1, 100,
                                                     size=(n_jobs, max(n_operations_for_jobs)))
        processing_time_matrix = -1 * np.ones((n_jobs, max(n_operations_for_jobs), n_machines))
        # n_compatible_machines_for_operations = np.random.randint(1, n_machines + 1,
        #                                                          size=(n_jobs, max(n_operations_for_jobs)))
        n_compatible_machines_for_operations = np.ones(shape=(n_jobs, max(n_operations_for_jobs)), dtype=int)

        for job_id in range(n_jobs):
            for operation_id in range(n_operations_for_jobs[job_id]):
                # processing_time_matrix[job_id, operation_id, :] = np.random.randint(
                #     max(1, processing_time_baseline[job_id, operation_id] - 10),
                #     min(100, processing_time_baseline[job_id, operation_id] + 11),
                #     size=(n_machines,)
                # )
                processing_time_matrix[job_id, operation_id, :] = processing_time_baseline[job_id, operation_id]
                zero_columns = np.random.choice(n_machines,
                                                n_machines - n_compatible_machines_for_operations[job_id, operation_id],
                                                replace=False)
                processing_time_matrix[job_id, operation_id, zero_columns] = -1

        empty_moving_time_matrix = np.hstack((layout_data_array[:n_machines, :n_machines], np.zeros((n_machines, 1))))
        empty_moving_time_matrix = np.vstack((empty_moving_time_matrix, np.zeros((1, n_machines + 1))))
        transport_time_matrix = empty_moving_time_matrix

        machine_quality = np.ones((1, n_machines), dtype=float)
        transbot_quality = np.ones((1, n_transbots), dtype=float)

        instance_with_quality = [
            n_operations_for_jobs.tolist(),
            job_arrival_time.tolist(),
            processing_time_matrix.tolist(),
            empty_moving_time_matrix.tolist(),
            transport_time_matrix.tolist(),
            machine_quality.tolist(),
            transbot_quality.tolist()
        ]
        instance_pool.append(instance_with_quality)

    return instance_pool

def generate_a_complete_instance(
    seed,
    max_n_jobs,
    max_n_machines,
    max_n_transbots,
) -> List:
    np.random.seed(seed)
    # n_jobs = max_n_jobs
    # n_jobs = np.random.randint(10, max_n_jobs + 1)
    n_jobs = 15
    n_machines = 15
    # n_machines = np.random.randint(5, max_n_machines + 1)
    # n_machines = 5 * np.random.randint(1, 4)
    # n_transbots = np.random.randint(10, max_n_transbots + 1)
    n_transbots = 2

    n_operations_for_jobs = n_machines * np.ones((n_jobs,), dtype=int)
    # if dfjspt_params.n_operations_is_n_machines:
    #     n_operations_for_jobs = n_machines * np.ones((n_jobs,), dtype=int)
    # else:
    #     n_operations_for_jobs = np.random.randint(dfjspt_params.min_n_operations, dfjspt_params.max_n_operations + 1,
    #                                               size=n_jobs, dtype=int)

    job_arrival_time = np.zeros(shape=(n_jobs,), dtype=int)

    processing_time_baseline = np.random.randint(1, 100,
                                                 size=(n_jobs, max(n_operations_for_jobs)))
    processing_time_matrix = -1 * np.ones((n_jobs, max(n_operations_for_jobs), n_machines))
    # n_compatible_machines_for_operations = np.random.randint(1, n_machines + 1,
    #                                                          size=(n_jobs, max(n_operations_for_jobs)))
    n_compatible_machines_for_operations = np.ones(shape=(n_jobs, max(n_operations_for_jobs)), dtype=int)

    for job_id in range(n_jobs):
        for operation_id in range(n_operations_for_jobs[job_id]):
            # processing_time_matrix[job_id, operation_id, :] = np.random.randint(
            #     max(1, processing_time_baseline[job_id, operation_id] - 10),
            #     min(100, processing_time_baseline[job_id, operation_id] + 11),
            #     size=(n_machines,)
            # )
            processing_time_matrix[job_id, operation_id, :] = processing_time_baseline[job_id, operation_id]
            zero_columns = np.random.choice(n_machines,
                                            n_machines - n_compatible_machines_for_operations[job_id, operation_id],
                                            replace=False)
            processing_time_matrix[job_id, operation_id, zero_columns] = -1

    empty_moving_time_matrix = np.hstack((layout_data_array[:n_machines, :n_machines], np.zeros((n_machines, 1))))
    empty_moving_time_matrix = np.vstack((empty_moving_time_matrix, np.zeros((1, n_machines + 1))))
    transport_time_matrix = empty_moving_time_matrix

    machine_quality = np.ones((1, n_machines), dtype=float)
    transbot_quality = np.ones((1, n_transbots), dtype=float)

    instance_with_quality = [
        n_operations_for_jobs.tolist(),
        job_arrival_time.tolist(),
        processing_time_matrix.tolist(),
        empty_moving_time_matrix.tolist(),
        transport_time_matrix.tolist(),
        machine_quality.tolist(),
        transbot_quality.tolist()
    ]

    return instance_with_quality


if __name__ == '__main__':

    single_instance = generate_a_complete_instance(
        seed=500,
        max_n_jobs=dfjspt_params.max_n_jobs,
        max_n_machines=dfjspt_params.max_n_machines,
        max_n_transbots=dfjspt_params.max_n_transbots,
    )

    my_instance_pool = generate_instance_pool(
        seed=dfjspt_params.instance_generator_seed,
        n_instances=dfjspt_params.n_instances,
        max_n_jobs=dfjspt_params.max_n_jobs,
        max_n_machines=dfjspt_params.max_n_machines,
        max_n_transbots=dfjspt_params.max_n_transbots,
    )

    folder_name = "sdata/J15_M15_T2"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    pool_name = folder_name + "/instance" + str(dfjspt_params.n_instances)

    with open(pool_name, "w") as fp:
        json.dump(my_instance_pool, fp)
    print(f"Data has been saved to {pool_name}")

    # with open("instance_pool", "r") as fp:
    #     loaded_pool = json.load(fp)
    #
    # print(loaded_pool)

