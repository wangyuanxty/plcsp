import numpy as np


def create_case_instance(n_job1, n_job2, n_job3, n_transbots):
    n_machines = 28

    # Travel time matrix data
    data = [
        [0, 2, 3.5, 4.5, 5, 4.5, 5, 5.5, 5, 4, 3, 3.5, 2.5, 3, 3, 2, 2.5, 4, 3, 4, 1.5, 0.5, 2, 1.5, 1, 3, 2.5, 2, 3, 2.5],
        [2, 0, 1.5, 2.5, 4, 4.5, 5, 5.5, 5, 4, 3, 3.5, 2.5, 3, 3, 2, 2.5, 4, 3, 4, 1.5, 2.5, 4, 3.5, 3, 5, 4.5, 4, 3, 2.5],
        [3.5, 1.5, 0, 1, 2.5, 3, 3.5, 4, 3.5, 2.5, 2.5, 2, 2, 1.5, 3.5, 3.5, 3, 2.5, 1.5, 2.5, 3, 4, 5.5, 5, 4.5, 6.5, 6, 5.5, 4.5, 4],
        [4.5, 2.5, 1, 0, 1.5, 2, 2.5, 3, 2.5, 3.5, 3.5, 3, 3, 2.5, 4.5, 4.5, 4, 1.5, 1.5, 1.5, 4, 5, 6.5, 6, 5.5, 7.5, 7, 6.5, 5.5, 5],
        [5, 4, 2.5, 1.5, 0, 0.5, 1, 1.5, 3, 4, 4, 3.5, 3.5, 3, 5, 5, 4.5, 2, 2, 1, 4.5, 5.5, 7, 6.5, 6, 8, 7.5, 7, 7, 6.5],
        [4.5, 4.5, 3, 2, 0.5, 0, 0.5, 1, 2.5, 3.5, 3.5, 3, 3, 2.5, 4.5, 4.5, 4, 1.5, 1.5, 0.5, 4, 5, 6.5, 6, 5.5, 7.5, 7, 6.5, 7.5, 7],
        [5, 5, 3.5, 2.5, 1, 0.5, 0, 0.5, 2, 3, 3, 2.5, 2.5, 2, 4, 4, 3.5, 1, 2, 1, 3.5, 4.5, 6, 5.5, 5, 7, 6.5, 7, 8, 7.5],
        [5.5, 5.5, 4, 3, 1.5, 1, 0.5, 0, 1.5, 2.5, 2.5, 2, 3, 2.5, 3.5, 3.5, 3, 1.5, 2.5, 1.5, 4, 5, 5.5, 6, 5.5, 6.5, 7, 7.5, 8.5, 8],
        [5, 5, 3.5, 2.5, 3, 2.5, 2, 1.5, 0, 1, 2, 1.5, 2.5, 2, 2, 3, 2.5, 1, 2, 2, 3.5, 4.5, 5, 5.5, 5, 6, 6.5, 7, 8, 7.5],
        [4, 4, 2.5, 3.5, 4, 3.5, 3, 2.5, 1, 0, 1, 0.5, 1.5, 1, 1, 2, 1.5, 2, 2, 3, 2.5, 3.5, 4, 4.5, 4, 5, 5.5, 6, 7, 6.5],
        [3, 3, 2.5, 3.5, 4, 3.5, 3, 2.5, 2, 1, 0, 0.5, 0.5, 1, 1, 1, 0.5, 2, 2, 3, 1.5, 2.5, 3, 3.5, 3, 4, 4.5, 5, 6, 5.5],
        [3.5, 3.5, 2, 3, 3.5, 3, 2.5, 2, 1.5, 0.5, 0.5, 0, 1, 0.5, 1.5, 1.5, 1, 1.5, 1.5, 2.5, 2, 3, 3.5, 4, 3.5, 4.5, 5, 5.5, 6.5, 6],
        [2.5, 2.5, 2, 3, 3.5, 3, 2.5, 3, 2.5, 1.5, 0.5, 1, 0, 0.5, 1.5, 1.5, 1, 1.5, 1.5, 2.5, 1, 2, 3.5, 3, 2.5, 4.5, 4, 4.5, 5.5, 5],
        [3, 3, 1.5, 2.5, 3, 2.5, 2, 2.5, 2, 1, 1, 0.5, 0.5, 0, 2, 2, 1.5, 1, 1, 2, 1.5, 2.5, 4, 3.5, 3, 5, 4.5, 5, 6, 5.5],
        [3, 3, 3.5, 4.5, 5, 4.5, 4, 3.5, 2, 1, 1, 1.5, 1.5, 2, 0, 1, 0.5, 3, 3, 4, 1.5, 2.5, 3, 3.5, 3, 4, 4.5, 5, 6, 5.5],
        [2, 2, 3.5, 4.5, 5, 4.5, 4, 3.5, 3, 2, 1, 1.5, 1.5, 2, 1, 0, 0.5, 3, 3, 4, 0.5, 1.5, 2, 2.5, 2, 3, 3.5, 4, 5, 4.5],
        [2.5, 2.5, 3, 4, 4.5, 4, 3.5, 3, 2.5, 1.5, 0.5, 1, 1, 1.5, 0.5, 0.5, 0, 2.5, 2.5, 3.5, 1, 2, 2.5, 3, 2.5, 3.5, 4, 4.5, 5.5, 5],
        [4, 4, 2.5, 1.5, 2, 1.5, 1, 1.5, 1, 2, 2, 1.5, 1.5, 1, 3, 3, 2.5, 0, 1, 1, 2.5, 3.5, 5, 4.5, 4, 6, 5.5, 6, 7, 6.5],
        [3, 3, 1.5, 1.5, 2, 1.5, 2, 2.5, 2, 2, 2, 1.5, 1.5, 1, 3, 3, 2.5, 1, 0, 1, 2.5, 3.5, 5, 4.5, 4, 6, 5.5, 5, 6, 5.5],
        [4, 4, 2.5, 1.5, 1, 0.5, 1, 1.5, 2, 3, 3, 2.5, 2.5, 2, 4, 4, 3.5, 1, 1, 0, 3.5, 4.5, 6, 5.5, 5, 7, 6.5, 6, 7, 6.5],
        [1.5, 1.5, 3, 4, 4.5, 4, 3.5, 4, 3.5, 2.5, 1.5, 2, 1, 1.5, 1.5, 0.5, 1, 2.5, 2.5, 3.5, 0, 1, 2.5, 2, 1.5, 3.5, 3, 3.5, 4.5, 4],
        [0.5, 2.5, 4, 5, 5.5, 5, 4.5, 5, 4.5, 3.5, 2.5, 3, 2, 2.5, 2.5, 1.5, 2, 3.5, 3.5, 4.5, 1, 0, 1.5, 1, 0.5, 2.5, 2, 2.5, 3.5, 3],
        [2, 4, 5.5, 6.5, 7, 6.5, 6, 5.5, 5, 4, 3, 3.5, 3.5, 4, 3.0, 2, 2.5, 5, 5, 6, 2.5, 1.5, 0, 0.5, 1, 1, 1.5, 2, 3, 2.5],
        [1.5, 3.5, 5, 6, 6.5, 6, 5.5, 6, 5.5, 4.5, 3.5, 4, 3, 3.5, 3.5, 2.5, 3, 4.5, 4.5, 5.5, 2, 1, 0.5, 0, 0.5, 1.5, 1, 1.5, 2.5, 2],
        [1, 3, 4.5, 5.5, 6, 5.5, 5, 5.5, 5, 4, 3, 3.5, 2.5, 3, 3, 2, 2.5, 4, 4, 5, 1.5, 0.5, 1, 0.5, 0, 2, 1.5, 2, 3, 2.5],
        [3, 5, 6.5, 7.5, 8, 7.5, 7, 6.5, 6, 5, 4, 4.5, 4.5, 5, 4, 3, 3.5, 6, 6, 7, 3.5, 2.5, 1, 1.5, 2, 0, 0.5, 1, 2, 2.5],
        [2.5, 4.5, 6, 7, 7.5, 7, 6.5, 7, 6.5, 5.5, 4.5, 5, 4, 4.5, 4.5, 3.5, 4, 5.5, 5.5, 6.5, 3, 2, 1.5, 1, 1.5, 0.5, 0, 0.5, 1.5, 2],
        [2, 4, 5.5, 6.5, 7, 6.5, 7, 7.5, 7, 6, 5, 5.5, 4.5, 5, 5, 4, 4.5, 6, 5, 6, 3.5, 2.5, 2, 1.5, 2, 1, 0.5, 0, 1, 1.5],
        [3, 3, 4.5, 5.5, 7, 7.5, 8, 8.5, 8, 7, 6, 6.5, 5.5, 6, 6, 5, 5.5, 7, 6, 7, 4.5, 3.5, 3, 2.5, 3, 2, 1.5, 1, 0, 0.5],
        [2.5, 2.5, 4, 5, 6.5, 7, 7.5, 8, 7.5, 6.5, 5.5, 6, 5, 5.5, 5.5, 4.5, 5, 6.5, 5.5, 6.5, 4, 3, 2.5, 2, 2.5, 2.5, 2, 1.5, 0.5, 0]
    ]

    matrix_without_first_row_and_column = np.array(data)[1:, 1:]
    last_row_and_column = matrix_without_first_row_and_column[:1, :]
    matrix_with_rearranged_first_row_and_column = np.vstack(
        (matrix_without_first_row_and_column[1:, :], last_row_and_column))
    last_column = matrix_with_rearranged_first_row_and_column[:, :1]
    matrix_with_rearranged_first_row_and_column = np.hstack(
        (matrix_with_rearranged_first_row_and_column[:, 1:], last_column))


    travel_time_matrix = matrix_with_rearranged_first_row_and_column

    # print("Travel Time Matrix:")
    # print(travel_time_matrix)

    n_operations_for_job1 = 29
    n_operations_for_job2 = 21
    n_operations_for_job3 = 29

    job1_matrix = np.full((n_operations_for_job1, n_machines), -1)
    job2_matrix = np.full((n_operations_for_job2, n_machines), -1)
    job3_matrix = np.full((n_operations_for_job3, n_machines), -1)

    job1_operations = {
        1: (1, 40), 2: (2, 50), 3: (3, 85), 4: (2, 20), 5: (4, 36),
        6: (5, 300), 7: (6, 145), 8: (7, 65), 9: (8, 50), 10: ((9, 6), (10, 6)),
        11: ((11, 150), (12, 150)), 12: (13, 50), 13: ((14, 3), (15, 3)),
        14: (16, 135), 15: (17, 120), 16: (18, 270), 17: (19, 100), 18: (20, 20),
        19: (21, 135), 20: (22, 80), 21: (23, 83), 22: ((24, 240), (25, 240), (26, 240)),
        23: (20, 15), 24: (27, 100), 25: (20, 15), 26: (20, 43), 27: (28, 33),
        28: (20, 20), 29: (28, 107)
    }

    for op, operations in job1_operations.items():
        if isinstance(operations[0], tuple):
            for operation in operations:
                machine, processing_time = operation
                job1_matrix[op - 1, machine - 1] = processing_time
        else:
            machine, processing_time = operations
            job1_matrix[op-1, machine-1] = processing_time

    job2_operations = {
        1: (3, 106), 2: (2, 20), 3: (17, 30), 4: (18, 270), 5: (8, 13),
        6: ((11, 180), (12, 180)), 7: (16, 34), 8: (22, 20), 9: (19, 25), 10: (20, 5),
        11: (21, 34), 12: (22, 20), 13: (23, 21), 14: ((24, 240), (25, 240), (26, 240)),
        15: (28, 8), 16: (20, 4), 17: (27, 25), 18: (20, 4), 19: (20, 11), 20: (20, 5),
        21: (28, 27)
    }

    for op, operations in job2_operations.items():
        if isinstance(operations[0], tuple):
            for operation in operations:
                machine, processing_time = operation
                job2_matrix[op - 1, machine - 1] = processing_time
        else:
            machine, processing_time = operations
            job2_matrix[op-1, machine-1] = processing_time

    job3_operations = {
        1: (1, 10), 2: (2, 50), 3: (3, 85), 4: (2, 20), 5: (4, 9),
        6: (5, 75), 7: (6, 145), 8: (7, 16), 9: (8, 13), 10: ((9, 6), (10, 6)),
        11: ((11, 150), (12, 150)), 12: (13, 13), 13: ((14, 3), (15, 3)),
        14: (16, 34), 15: (17, 30), 16: (18, 270), 17: (19, 25), 18: (20, 5),
        19: (21, 34), 20: (22, 20), 21: (23, 21), 22: ((24, 240), (25, 240), (26, 240)),
        23: (28, 8), 24: (20, 4), 25: (27, 25), 26: (20, 4), 27: (20, 11), 28: (20, 5),
        29: (28, 27)
    }

    for op, operations in job3_operations.items():
        if isinstance(operations[0], tuple):
            for operation in operations:
                machine, processing_time = operation
                job3_matrix[op - 1, machine - 1] = processing_time
        else:
            machine, processing_time = operations
            job3_matrix[op-1, machine-1] = processing_time

    # print("\nJob 1 Matrix:")
    # print(job1_matrix)
    # print("\nJob 2 Matrix:")
    # print(job2_matrix)
    # print("\nJob 3 Matrix:")
    # print(job3_matrix)

    n_operations_for_jobs = np.concatenate((np.full(n_job1, n_operations_for_job1), np.full(n_job2, n_operations_for_job2), np.full(n_job3, n_operations_for_job3)))
    job_arrival_time = np.zeros(shape=(n_job1+n_job2+n_job3,), dtype=float)
    machine_quality = np.ones((1, n_machines), dtype=float)
    transbot_quality = np.ones((1, n_transbots), dtype=float)

    processing_time_matrix = np.zeros((n_job1+n_job2+n_job3, max(n_operations_for_job1,n_operations_for_job2,n_operations_for_job3), n_machines))
    matrices = [job1_matrix] * n_job1 + [job2_matrix] * n_job2 + [job3_matrix] * n_job3
    current_row = 0
    for matrix in matrices:
        processing_time_matrix[current_row][:matrix.shape[0], :] = matrix
        current_row += 1


    case_instance = [n_operations_for_jobs, job_arrival_time, processing_time_matrix,
                     travel_time_matrix, travel_time_matrix, machine_quality, transbot_quality]

    return case_instance

if __name__ == '__main__':

    n_job1 = 1
    n_job2 = 4
    n_job3 = 4
    n_transbots = 2

    case_instance = create_case_instance(n_job1, n_job2, n_job3, n_transbots)

    print(case_instance)