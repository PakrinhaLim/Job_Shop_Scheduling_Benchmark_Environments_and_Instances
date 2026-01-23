# DANIEL Model Explanation

This document explains the **DANIEL** model (Deep Attention Network for fIexible job shop scheduLing) found in the `solution_methods/DANIEL` directory.

## Overview

DANIEL is a **Deep Reinforcement Learning (DRL)** approach that uses a **Dual Attention Network (DAN)** to solve the Flexible Job Shop Scheduling Problem (FJSP). It employs **Proximal Policy Optimization (PPO)** to learn a policy that selects the best operation-machine pair at each decision step.

## 1. Model Architecture

The model architecture is defined primarily in `network/main_model.py` and `network/PPO.py`.

### Dual Attention Network (DAN)
The core feature extractor is the `DualAttentionNetwork`, which updates embeddings for **Operations** and **Machines** through stacked attention layers.

*   **Operation Attention Block (OAB)**:
    *   Defined in `network/attention_layer.py` (`MultiHeadOpAttnBlock`).
    *   Updates operation embeddings by aggregating information from their precedence relations (predecessors and successors) using a masked self-attention mechanism.
*   **Machine Attention Block (MAB)**:
    *   Defined in `network/attention_layer.py` (`MultiHeadMchAttnBlock`).
    *   Updates machine embeddings by considering the "competition" between machines.
    *   It uses a competition matrix (`comp_val` or $c_{kq}$) derived from the features of operations that are candidates for these machines.

The network stacks multiple layers (defined by `num_dan_layers`) where OAB and MAB are applied sequentially.

### Actor-Critic Network
The system uses an Actor-Critic architecture wrapped in the `DANIEL` class (`network/main_model.py`):

*   **Actor**:
    *   **Input**: Concatenation of:
        *   Candidate Operation features (Job)
        *   Candidate Machine features
        *   Global (pooled) Operation features
        *   Global (pooled) Machine features
        *   Pair features (specific to the operation-machine pair)
    *   **Output**: A probability distribution over all valid (Job, Machine) pairs.
    *   **Mechanism**: Uses an MLP to score pairs, masks invalid ones (incompatible machine, job not ready), and applies Softmax.
*   **Critic**:
    *   **Input**: Concatenation of Global Operation features and Global Machine features.
    *   **Output**: A scalar Value estimate.

## 2. Training Process (`train_DANIEL.py`)

The training script uses PPO to train the DANIEL model.

*   **Configuration**: Parameters loaded from `configs/DANIEL.toml`.
*   **Environment**: Uses `FJSPEnvForSameOpNums` or `FJSPEnvForVariousOpNums` (in `src/`).
*   **Data Generation**:
    *   Generates random instances on the fly using `CaseGenerator` or `SD2_instance_generator`.
    *   Supports two data sources: "SD1" (classic) and "SD2" (sequence dependent, if configured).
*   **Training Loop**:
    1.  **Rollout**: Collects trajectories using the current policy.
    2.  **Storage**: Stores states, actions, rewards, etc., in `Memory`.
    3.  **Update**: Every `max_updates` (or when memory is full/episode done), it updates the policy using PPO.
        *   Calculates GAE (Generalized Advantage Estimation).
        *   Optimizes the surrogate objective function (Actor loss) and Value loss (Critic loss).
    4.  **Validation**: Periodically validates on a fixed dataset. Saves the model if performance improves.

## 3. Running / Inference (`run_DANIEL.py`)

The running script loads a trained model to solve specific instances.

*   **Model Loading**: Loads the `.pth` checkpoint specified in `test_parameters`.
*   **Environment**: `FJSPEnv_test`.
*   **Execution**:
    *   Iteratively calls `ppo.policy` to get action probabilities.
    *   Selects actions using **Greedy** (argmax) or **Sampling** strategy.
    *   Steps the environment until all jobs are completed.
*   **Output**:
    *   Logs the final **Makespan**.
    *   Can generate Gantt charts and save execution results.

## Summary of Files

| File | Description |
| :--- | :--- |
| `train_DANIEL.py` | Main script for training the model. |
| `run_DANIEL.py` | Main script for testing/running values. |
| `network/main_model.py` | Defines `DANIEL` (Actor-Critic) and `DualAttentionNetwork`. |
| `network/PPO.py` | Defines the `PPO` algorithm logic and `Memory` buffer. |
| `network/attention_layer.py`| Defines the OAB and MAB attention blocks. |
| `src/fjsp_env_*.py` | Training environments (Same/Various Op Num). |
