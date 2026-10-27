import os
import numpy as np
from typing import List
from plcsp_v4 import dfjspt_params
import json


def generate_layout(
        layout_seed=dfjspt_params.layout_seed,
):
    np.random.seed(layout_seed)
    layout = np.random.randint(low=dfjspt_params.min_tspt_time,
                               high=dfjspt_params.max_tspt_time,
                               size=(dfjspt_params.max_n_machines + 1, dfjspt_params.max_n_machines + 1))
    layout = np.triu(layout, k=1)
    layout += layout.T
    return layout


def generate_instance_pool(
        seed,
        n_instances,
        n_jobs,
        n_machines,
        n_transbots,
) -> List:
    instance_pool = []
    # np.random.seed(seed)

    for i in range(n_instances):
        np.random.seed(seed + i)

        if dfjspt_params.n_jobs_is_fixed:
            n_jobs_for_this_instance = n_jobs
        else:
            n_jobs_for_this_instance = np.random.randint(10, n_jobs + 1)

        if dfjspt_params.n_operations_is_n_machines:
            n_operations_for_jobs = n_machines * np.ones((n_jobs_for_this_instance,), dtype=int)
        else:
            n_operations_for_jobs = np.random.randint(low=dfjspt_params.min_n_operations,
                                                      high=dfjspt_params.max_n_operations + 1,
                                                      size=n_jobs_for_this_instance, dtype=int)

        job_arrival_time = np.zeros(shape=(n_jobs,), dtype=int)
        if dfjspt_params.consider_job_insert and dfjspt_params.new_arrival_jobs > 0:
            for new_job in range(dfjspt_params.new_arrival_jobs):
                job_arrival_time[n_jobs - new_job - 1] = np.random.randint(
                    low=dfjspt_params.earliest_arrive_time,
                    high=dfjspt_params.latest_arrive_time + 1
                )

        processing_time_baseline = np.random.randint(low=dfjspt_params.min_prcs_time,
                                                     high=dfjspt_params.max_prcs_time,
                                                     size=(n_jobs, max(n_operations_for_jobs)))
        processing_time_matrix = -1 * np.ones((n_jobs, max(n_operations_for_jobs), n_machines))
        if dfjspt_params.is_fully_flexible:
            n_compatible_machines_for_operations = n_machines * np.ones(shape=(n_jobs, max(n_operations_for_jobs)),
                                                                        dtype=int)
        else:
            n_compatible_machines_for_operations = np.random.randint(1, n_machines + 1,
                                                                     size=(n_jobs, max(n_operations_for_jobs)))

        for job_id in range(n_jobs):
            for operation_id in range(n_operations_for_jobs[job_id]):
                if dfjspt_params.time_for_compatible_machines_are_same:
                    processing_time_matrix[job_id, operation_id, :] = processing_time_baseline[job_id, operation_id]
                else:
                    processing_time_matrix[job_id, operation_id, :] = np.random.randint(
                        max(dfjspt_params.min_prcs_time,
                            processing_time_baseline[job_id, operation_id] - dfjspt_params.time_viration_range),
                        min(dfjspt_params.max_prcs_time,
                            processing_time_baseline[job_id, operation_id] + dfjspt_params.time_viration_range + 1),
                        size=(n_machines,)
                    )
                zero_columns = np.random.choice(n_machines,
                                                n_machines - n_compatible_machines_for_operations[job_id, operation_id],
                                                replace=False)
                processing_time_matrix[job_id, operation_id, zero_columns] = -1

        if dfjspt_params.all_machines_are_perfect:
            machine_quality = np.ones((1, n_machines), dtype=float)
            transbot_quality = np.ones((1, n_transbots), dtype=float)
        else:
            num_perfect_machine = np.random.randint(0, n_machines + 1)
            orig_machine_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_machines,)), 1)
            machine_mask = np.random.choice(n_machines, num_perfect_machine, replace=False)
            orig_machine_quality[machine_mask] = 1.0
            machine_quality = orig_machine_quality.reshape((1, n_machines))

            num_perfect_transbot = np.random.randint(0, n_transbots + 1)
            orig_transbot_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_transbots,)), 1)
            transbot_mask = np.random.choice(n_transbots, num_perfect_transbot, replace=False)
            orig_transbot_quality[transbot_mask] = 1.0
            transbot_quality = orig_transbot_quality.reshape((1, n_transbots))

        empty_moving_time_matrix = generate_layout()[:n_machines + 1, :n_machines + 1]
        transport_time_matrix = empty_moving_time_matrix * dfjspt_params.loaded_transport_time_scale

        instance_with_quality = [
            n_operations_for_jobs.tolist(),
            job_arrival_time.tolist(),
            processing_time_matrix.tolist(),
            empty_moving_time_matrix.tolist(),
            transport_time_matrix.tolist(),
            machine_quality.tolist(),
            transbot_quality.tolist()]

        instance_pool.append(instance_with_quality)

    return instance_pool

def generate_a_complete_instance(
    seed,
    n_jobs,
    n_machines,
    n_transbots,
) -> List:

    np.random.seed(seed)

    if dfjspt_params.n_jobs_is_fixed:
        n_jobs_for_this_instance = n_jobs
    else:
        n_jobs_for_this_instance = np.random.randint(10, n_jobs + 1)

    if dfjspt_params.n_operations_is_n_machines:
        n_operations_for_jobs = n_machines * np.ones((n_jobs_for_this_instance,), dtype=int)
    else:
        n_operations_for_jobs = np.random.randint(low=dfjspt_params.min_n_operations,
                                                  high=dfjspt_params.max_n_operations + 1,
                                                  size=n_jobs_for_this_instance, dtype=int)

    job_arrival_time = np.zeros(shape=(n_jobs,), dtype=int)
    if dfjspt_params.consider_job_insert and dfjspt_params.new_arrival_jobs > 0:
        for new_job in range(dfjspt_params.new_arrival_jobs):
            job_arrival_time[n_jobs - new_job - 1] = np.random.randint(
                low=dfjspt_params.earliest_arrive_time,
                high=dfjspt_params.latest_arrive_time + 1
            )

    processing_time_baseline = np.random.randint(low=dfjspt_params.min_prcs_time,
                                                 high=dfjspt_params.max_prcs_time,
                                                 size=(n_jobs, max(n_operations_for_jobs)))
    processing_time_matrix = -1 * np.ones((n_jobs, max(n_operations_for_jobs), n_machines))
    if dfjspt_params.is_fully_flexible:
        n_compatible_machines_for_operations = n_machines * np.ones(shape=(n_jobs, max(n_operations_for_jobs)),
                                                                    dtype=int)
    else:
        n_compatible_machines_for_operations = np.random.randint(1, n_machines + 1,
                                                                 size=(n_jobs, max(n_operations_for_jobs)))

    for job_id in range(n_jobs):
        for operation_id in range(n_operations_for_jobs[job_id]):
            if dfjspt_params.time_for_compatible_machines_are_same:
                processing_time_matrix[job_id, operation_id, :] = processing_time_baseline[job_id, operation_id]
            else:
                processing_time_matrix[job_id, operation_id, :] = np.random.randint(
                    max(dfjspt_params.min_prcs_time, processing_time_baseline[job_id, operation_id] - dfjspt_params.time_viration_range),
                    min(dfjspt_params.max_prcs_time, processing_time_baseline[job_id, operation_id] + dfjspt_params.time_viration_range + 1),
                    size=(n_machines,)
                )
            zero_columns = np.random.choice(n_machines,
                                            n_machines - n_compatible_machines_for_operations[job_id, operation_id],
                                            replace=False)
            processing_time_matrix[job_id, operation_id, zero_columns] = -1

    if dfjspt_params.all_machines_are_perfect:
        machine_quality = np.ones((1, n_machines), dtype=float)
        transbot_quality = np.ones((1, n_transbots), dtype=float)
    else:
        num_perfect_machine = np.random.randint(0, n_machines + 1)
        orig_machine_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_machines,)), 1)
        machine_mask = np.random.choice(n_machines, num_perfect_machine, replace=False)
        orig_machine_quality[machine_mask] = 1.0
        machine_quality = orig_machine_quality.reshape((1, n_machines))

        num_perfect_transbot = np.random.randint(0, n_transbots + 1)
        orig_transbot_quality = np.round(np.random.uniform(dfjspt_params.min_quality, 1.0, size=(n_transbots,)), 1)
        transbot_mask = np.random.choice(n_transbots, num_perfect_transbot, replace=False)
        orig_transbot_quality[transbot_mask] = 1.0
        transbot_quality = orig_transbot_quality.reshape((1, n_transbots))

    empty_moving_time_matrix = generate_layout()[:n_machines + 1, :n_machines + 1]
    transport_time_matrix = empty_moving_time_matrix * dfjspt_params.loaded_transport_time_scale

    np.random.seed(seed)

    instance_with_quality = [
        n_operations_for_jobs.tolist(),
        job_arrival_time.tolist(),
        processing_time_matrix.tolist(),
        empty_moving_time_matrix.tolist(),
        transport_time_matrix.tolist(),
        machine_quality.tolist(),
        transbot_quality.tolist()]

    return instance_with_quality




if __name__ == '__main__':

    my_instance_pool = generate_instance_pool(
        seed=dfjspt_params.instance_generator_seed,
        n_instances=dfjspt_params.n_instances,
        n_jobs=dfjspt_params.n_jobs,
        n_machines=dfjspt_params.n_machines,
        n_transbots=dfjspt_params.n_transbots,
    )

    folder_name = os.path.dirname(__file__) + "/my_data/pool_J" + str(dfjspt_params.n_jobs) + "_M" + str(dfjspt_params.n_machines) + "_T" + str(dfjspt_params.n_transbots)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    pool_name = folder_name + "/instance" + str(dfjspt_params.n_instances) + "_Omin" + str(dfjspt_params.min_n_operations) + "_Omax" + str(dfjspt_params.max_n_operations)

    with open(pool_name, "w") as fp:
        json.dump(my_instance_pool, fp)
    print(f"Data has been saved to {pool_name}")

    # with open("instance_pool", "r") as fp:
    #     loaded_pool = json.load(fp)
    #
    # print(loaded_pool)

