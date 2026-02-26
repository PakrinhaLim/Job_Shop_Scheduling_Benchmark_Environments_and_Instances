import random
import os
import json

from visualization import gantt_chart, precedence_chart
from data.data_parsers.parser_fjsp import parse_fjsp
from scheduling_environment.jobShop import JobShop

from solution_methods.helper_functions import load_job_shop_env, load_parameters
from solution_methods.dispatching_rules.run_dispatching_rules import run_dispatching_rules
from solution_methods.GA.src.initialization import initialize_run
from solution_methods.GA.run_GA import run_GA
from solution_methods.MILP.run_MILP import run_MILP
from solution_methods.FJSP_DRL.run_FJSP_DRL import run_FJSP_DRL



# fjsp_file_paths = [os.path.join(root, file).replace('data', '').replace("\\", "/") for root, dirs, files in os.walk('data/fjsp') for file in files]
fjsp_file_paths = [os.path.join(root, file).replace('data', '', 1).replace("\\", "/") for root, dirs, files in os.walk('data/fjsp/hurink') for file in files]

# Load data from file
# jobShopEnv = load_job_shop_env(fjsp_file_paths[0])
# jobShopEnv.update_operations_available_for_scheduling()

# while len(jobShopEnv.scheduled_operations) < jobShopEnv.nr_of_operations:
#     operation = random.choice(jobShopEnv.operations_available_for_scheduling)
#     machine_id = random.choice(list(operation.processing_times.keys()))
#     duration = operation.processing_times[machine_id]
#     jobShopEnv.schedule_operation_on_machine(operation, machine_id, duration)
#     jobShopEnv.update_operations_available_for_scheduling()

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
total_instance = len(fjsp_file_paths)


for i, fjsp_file_path in enumerate(fjsp_file_paths):
    result_path = os.path.join("results/plots", os.path.basename(fjsp_file_path).replace(".fjs", ".png"))
    if os.path.exists(result_path):
        continue

    jobShopEnv = load_job_shop_env(fjsp_file_path)
    makespan, jobShopEnv = run_FJSP_DRL(jobShopEnv, **parameters)
    print("{} / {} | {} | Makespan: {}".format(i + 1, total_instance, fjsp_file_path.split('/')[-1], makespan))

    os.makedirs("results/json", exist_ok=True)
    with open(result_path, "w") as f:
        json.dump({"instance": fjsp_file_path, "makespan": makespan}, f, indent=4)

    plt = gantt_chart.plot(jobShopEnv)
    output_dir = "results/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, os.path.basename(fjsp_file_path).replace(".fjs", ".png")))
    plt.close()


# plt = gantt_chart.plot(jobShopEnv)
# plt.show()

# output_dir = "results/plots"
# os.makedirs(output_dir, exist_ok=True)
# plt.savefig(os.path.join(output_dir, "gantt_chart.png"))
