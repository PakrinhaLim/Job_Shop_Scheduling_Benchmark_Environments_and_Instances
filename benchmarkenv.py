import random

from visualization import gantt_chart, precedence_chart
from data.data_parsers.parser_fjsp import parse_fjsp
from scheduling_environment.jobShop import JobShop

from solution_methods.helper_functions import load_job_shop_env, load_parameters
from solution_methods.dispatching_rules.run_dispatching_rules import run_dispatching_rules
from solution_methods.GA.src.initialization import initialize_run
from solution_methods.GA.run_GA import run_GA
from solution_methods.MILP.run_MILP import run_MILP
from solution_methods.FJSP_DRL.run_FJSP_DRL import run_FJSP_DRL

jobShopEnv = JobShop()
# jobShopEnv = parse_fjsp(jobShopEnv, '/fjsp/brandimarte/MK09.fjs')
# jobShopEnv = load_job_shop_env('/fjsp/brandimarte/MK09.fjs')


# print(jobShopEnv)

import os

fjsp_file_paths = [os.path.join(root, file).replace('data', '') for root, dirs, files in os.walk('data/fjsp') for file in files]
# fjsp_file_paths = []
# for root, dirs, files in os.walk('data/fjsp'):
#     for file in files:
#         if file.endswith('.fjs'):
#             path = os.path.join(root, file).replace('data', '')
#             fjsp_file_paths.append(path)
print(fjsp_file_paths)

# jobShopEnv = load_job_shop_env('/fjsp/brandimarte/MK09.fjs')
jobShopEnv = load_job_shop_env(fjsp_file_paths[0])
jobShopEnv.update_operations_available_for_scheduling()

while len(jobShopEnv.scheduled_operations) < jobShopEnv.nr_of_operations:
    operation = random.choice(jobShopEnv.operations_available_for_scheduling)
    machine_id = random.choice(list(operation.processing_times.keys()))
    duration = operation.processing_times[machine_id]
    jobShopEnv.schedule_operation_on_machine(operation, machine_id, duration)
    jobShopEnv.update_operations_available_for_scheduling()

# precedence_chart.plot(jobShopEnv)
# plt = gantt_chart.plot(jobShopEnv)
# plt.show()

# # Dispatching Rules
# parameters = load_parameters("configs/dispatching_rules.toml")
# # jobShopEnv = load_job_shop_env(parameters['instance'].get('problem_instance'))
# jobShopEnv = load_job_shop_env('/fjsp/brandimarte/MK09.fjs')

# makespan, jobShopEnv = run_dispatching_rules(jobShopEnv, **parameters)
# print("makespan: ", makespan)

parameters = load_parameters("configs/fjsp_drl.toml")
jobShopEnv = load_job_shop_env(parameters['test_parameters'].get('problem_instance'))

makespan, jobShopEnv = run_FJSP_DRL(jobShopEnv, **parameters)

plt = gantt_chart.plot(jobShopEnv)
plt.show()