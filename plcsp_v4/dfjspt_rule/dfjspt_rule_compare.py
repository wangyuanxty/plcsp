import json
import os
import time
import numpy as np
from plcsp_v4.dfjspt_rule.dfjspt_rule1_EST_EET_EET import rule1_mean_makespan, rule1_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule2_EST_SPT_SPT import rule2_mean_makespan, rule2_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule3_MOPNR_EET_EET import rule3_mean_makespan, rule3_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule4_MOPNR_SPT_SPT import rule4_mean_makespan, rule4_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule5_SPT_EET_EET import rule5_mean_makespan, rule5_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule6_SPT_SPT_SPT import rule6_mean_makespan, rule6_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule7_MTWR_EET_EET import rule7_mean_makespan, rule7_single_makespan
from plcsp_v4.dfjspt_rule.dfjspt_rule8_MTWR_SPT_SPT import rule8_mean_makespan, rule8_single_makespan
from plcsp_v4 import dfjspt_params


rules_average = {
    1: rule1_mean_makespan,
    2: rule2_mean_makespan,
    3: rule3_mean_makespan,
    4: rule4_mean_makespan,
    5: rule5_mean_makespan,
    6: rule6_mean_makespan,
    7: rule7_mean_makespan,
    8: rule8_mean_makespan,
}

rules_single = {
    1: rule1_single_makespan,
    2: rule2_single_makespan,
    3: rule3_single_makespan,
    4: rule4_single_makespan,
    5: rule5_single_makespan,
    6: rule6_single_makespan,
    7: rule7_single_makespan,
    8: rule8_single_makespan,
}

average_makespans = np.zeros((8,))
num_repeat = 10

print(f"Start testing instances with {dfjspt_params.n_jobs} jobs, {dfjspt_params.n_machines} machines, and {dfjspt_params.n_transbots} transbots using heuristic dispatching rules.")

for i in range(1, 9):
    # print(f"rule {i}:")
    rule_single = rules_single.get(i)
    if rule_single:
        makespan_list = []

        for j in range(dfjspt_params.n_instances_for_testing):
            makespan_repeat = []
            for repeat in range(num_repeat):
                makespan_k = rule_single(
                    instance_id=j,
                    train_or_eval_or_test="test",
                )
                makespan_repeat.append(makespan_k)
            makespan = np.mean(makespan_repeat)

            makespan_list.append(makespan)
        average_makespan = np.mean(makespan_list)
        print(f"Rule{i}: Average makespans for each test instance are: \n")
        print(makespan_list)
        print(f"Total average makespan across test instances is {average_makespan}\n")
        average_makespans[i-1] = average_makespan

best_rule = np.argmin(average_makespans) + 1
print(f"The best rule in J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots} is rule {best_rule}, average makespan is {average_makespans[best_rule-1]}")

