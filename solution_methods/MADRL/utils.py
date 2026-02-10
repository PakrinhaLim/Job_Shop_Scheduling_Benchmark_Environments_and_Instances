import os
import datetime
import json

DEFAULT_RESULTS_ROOT = os.path.join(os.getcwd(), "results", "MADRL")

def output_dir_exp_name(parameters):
    test_params = parameters.get('test_parameters', {})
    
    if test_params.get('exp_name'):
        exp_name = test_params['exp_name']
    else:
        instance_name = test_params.get('problem_instance', 'unknown').replace('/', '_')
        if instance_name.startswith('_'):
            instance_name = instance_name[1:]
        
        trained_policy = test_params.get('trained_policy', 'default')
        network = os.path.basename(trained_policy).split('.')[0]
        
        strategy = 'sample' if test_params.get('sample') else 'greedy'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{instance_name}_network_{network}_{strategy}_{timestamp}"

    if test_params.get('folder'):
        output_dir = test_params['folder']
    else:
        output_dir = DEFAULT_RESULTS_ROOT
        
    return output_dir, exp_name

def results_saving(makespan, jobShopEnv, path, parameters):
    """
    Save the MADRL results to a JSON file.
    """
    
    schedule = []
    # jobShopEnv.jobs should have the scheduled operations
    for job in jobShopEnv.jobs:
        job_info = {"job": job.job_id, "tasks": []}
        for op in job.operations:
            if op.scheduled_machine is not None:
                task_info = {
                    "task": op.operation_id,
                    "start": op.scheduled_start_time,
                    "machine": op.scheduled_machine,
                    "duration": op.scheduled_duration,
                    "end": op.scheduled_end_time
                }
                job_info["tasks"].append(task_info)
        schedule.append(job_info)

    results = {
        "instance": parameters["test_parameters"]["problem_instance"],
        "makespan": makespan,
        "trained_policy" : parameters['test_parameters']['trained_policy'],
        "sample": parameters['test_parameters']['sample'],
        "seed": parameters['test_parameters']['seed'],
        "schedule": schedule
    }

    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "MADRL_results.json")
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
