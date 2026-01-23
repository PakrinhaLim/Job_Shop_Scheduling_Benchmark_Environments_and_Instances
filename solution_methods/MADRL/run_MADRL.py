
import argparse
import logging
import os
import torch
import sys
from pathlib import Path

# Add project root to path
base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

from visualization import gantt_chart, precedence_chart
from solution_methods.helper_functions import load_job_shop_env, load_parameters, initialize_device, set_seeds
from solution_methods.DANIEL.src.common_utils import sample_action
from solution_methods.MADRL.src.env_test_madrl import MADRL_FJSPEnv_test
from solution_methods.MADRL.network.ppo_madrl import PPO_initialize
from solution_methods.DANIEL.utils import output_dir_exp_name, results_saving
from solution_methods.MADRL.train_MADRL import sample_action_madrl

# Re-use DANIEL config structure where possible
PARAM_FILE = "configs/MADRL.toml"
logging.basicConfig(level=logging.INFO)

def run_MADRL_FJSP(jobShopEnv, **parameters):
    # Set up device and seeds
    device = initialize_device(parameters, method="DANIEL")
    set_seeds(parameters["test_parameters"]["seed"])

    # Configure default device
    if device.type == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Configure test environment
    env_test = MADRL_FJSPEnv_test(jobShopEnv, parameters)

    # Initialize PPO model
    ppo = PPO_initialize(parameters)

    # load trained policy
    model_source = parameters['model']['source']
    trained_policy_name = parameters['test_parameters']['trained_policy']
    trained_policy_path = (os.path.dirname(os.path.abspath(__file__)) + f"\save\{model_source}\{trained_policy_name}.pth")
    
    if not os.path.exists(trained_policy_path):
        logging.error(f"Trained policy not found at {trained_policy_path}")
        return None, None

    policy = torch.load(trained_policy_path, map_location=device, weights_only=True)
    ppo.policy.load_state_dict(policy)
    ppo.policy.eval()
    logging.info(f"Trained policy loaded from {trained_policy_path}.")

    # Get state
    state = env_test.state

    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
            )

        # Choose action
        # pi: [1, J, M]
        # We need to sample actions or take greedy max
        if parameters["test_parameters"]["sample"]:
            actions, _ = sample_action_madrl(pi) # Returns [1, M]
        else:
            # Greedy: argmax over J for each M
            # Need to handle invalid actions (where pi is all masked/0)
            # pi has -inf or nan where invalid? 
            # In madrl_model: pi = softmax(scores). Compatible pairs only. 
            # Incompatible are masked to -inf before softmax -> 0 probability.
            # So we can just take argmax.
            # But if a machine has NO valid jobs, all are 0/nan.
            
            # Check for invalid machines (all prob 0 or nan)
            # pi [1, J, M]
            pi_m = pi.permute(0, 2, 1) # [1, M, J]
            actions = torch.argmax(pi_m, dim=2) # [1, M]
            
            # If all are perfectly 0, argmax might just pick index 0. 
            # In validation/test, strict greedy is fine unless it picks an invalid op.
            # MADRL environment checks validity. If we pick an invalid op, env might complain or crash?
            # Env checks valid_job_indices.
            # Let's ensure we pick a valid one or -1.
            
            # Identify machines with no valid options
            # If dynamic_pair_mask is True (invalid), then score was -inf.
            # dynamic_pair_mask: [sz_b, J, M]
            # If all J are True for a machine m, then m has no actions.
            
            mask = state.dynamic_pair_mask_tensor # [1, J, M]
            all_invalid = mask.all(dim=1) # [1, M]
            
            actions[all_invalid] = -1
            
        # Perform action
        state, reward, done = env_test.step_madrl(actions.cpu().numpy())

        if done.all():
            break

    makespan = env_test.JSP_instance.makespan
    logging.info(f"Makespan: {makespan}")

    return makespan, env_test.JSP_instance


def main(param_file=PARAM_FILE):
    try:
        parameters = load_parameters(param_file)
    except FileNotFoundError:
        logging.error(f"Parameter file {param_file} not found.")
        return

    jobShopEnv = load_job_shop_env(parameters["test_parameters"]["problem_instance"])
    makespan, jobShopEnv = run_MADRL_FJSP(jobShopEnv, **parameters)

    if makespan is not None:
        # Check output configuration and prepare output paths if needed
        output_config = parameters['test_parameters']
        save_gantt = output_config.get('save_gantt')
        save_results = output_config.get('save_results')
        show_gantt = output_config.get('show_gantt')
        show_precedences = output_config.get('show_precedences')

        if save_gantt or save_results:
            output_dir, exp_name = output_dir_exp_name(parameters)
            output_dir = os.path.join(output_dir, f"{exp_name}")
            os.makedirs(output_dir, exist_ok=True)

        # Draw precedence relations if required
        if show_precedences:
            precedence_chart.plot(jobShopEnv)

        # Plot Gantt chart if required
        if show_gantt or save_gantt:
            logging.info("Generating Gantt chart.")
            plt = gantt_chart.plot(jobShopEnv)

            if save_gantt:
                plt.savefig(output_dir + "/gantt.png")
                logging.info(f"Gantt chart saved to {output_dir}")

            if show_gantt:
                plt.show()

        # Save results if enabled
        if save_results:
            results_saving(makespan, output_dir, parameters)
            logging.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MADRL")
    parser.add_argument(
        "config_file",
        metavar="-f",
        type=str,
        nargs="?",
        default=PARAM_FILE,
        help="path to config file",
    )

    args = parser.parse_args()
    main(param_file=args.config_file)
