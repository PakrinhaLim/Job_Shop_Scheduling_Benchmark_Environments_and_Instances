import random
import numpy as np

import tomli
import torch

import json
from data.data_parsers import parser_fjsp, parser_fajsp, parser_fjsp_sdst, parser_jsp_fsp, custom_instance_parser
from scheduling_environment.jobShop import JobShop


def load_parameters(config_toml):
    """Load parameters from a toml file"""
    with open(config_toml, "rb") as f:
        config_params = tomli.load(f)
    return config_params


def load_job_shop_env(problem_instance: str, from_absolute_path=False) -> JobShop:
    jobShopEnv = JobShop()
    if problem_instance.endswith('.json'):
        if from_absolute_path:
             path = problem_instance
        else:
             # Assuming standard path structure if needed, or just fail if not found
             # helper_functions usually assumes paths relative to project root or similar if from_absolute_path=False
             # But let's assume if it is .json it might be a direct path or relative path the user provided.
             # The existing parsers handle paths internally.
             # Let's try to open it directly if from_absolute_path is True, else constructing path might be tricky without base dir.
             # However, run_DANIEL.py passes parameters["test_parameters"]["problem_instance"].
             # Let's enforce absolute path usage or simple relative path open for now.
             path = problem_instance

        with open(path, 'r') as f:
             data = json.load(f)
             # Support both direct structure or wrapped in processing_info
             if "processing_info" in data:
                 processing_info = data["processing_info"]
             else:
                 processing_info = data
        jobShopEnv = custom_instance_parser.parse(processing_info, instance_name=problem_instance)

    elif '/fsp/' in problem_instance or '/jsp/' in problem_instance:
        jobShopEnv = parser_jsp_fsp.parse_jsp_fsp(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fjsp/' in problem_instance:
        jobShopEnv = parser_fjsp.parse_fjsp(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fjsp_sdst/' in problem_instance:
        jobShopEnv = parser_fjsp_sdst.parse_fjsp_sdst(jobShopEnv, problem_instance, from_absolute_path)
    elif '/fajsp/' in problem_instance:
        jobShopEnv = parser_fajsp.parse_fajsp(jobShopEnv, problem_instance, from_absolute_path)
    else:
        raise NotImplementedError(
            f"""Problem instance {
            problem_instance
            } not implemented"""
        )
    jobShopEnv._name = problem_instance
    return jobShopEnv


def set_seeds(seed_value=0):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def initialize_device(parameters: dict, method: str = "FJSP_DRL") -> torch.device:
    device_str = "cpu"
    if method == "FJSP_DRL":
        if parameters['test_parameters']['device'] == "cuda":
            device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif method == "DANIEL":
        if parameters["device"]["name"] == "cuda":
            device_str = (
                f"cuda:{parameters['device']['id']}" if torch.cuda.is_available() else "cpu"
            )
    return torch.device(device_str)