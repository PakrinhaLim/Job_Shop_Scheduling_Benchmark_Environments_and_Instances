# FJSP_DRL Model Explanation

This document explains the **FJSP_DRL** (Flexible Job Shop Scheduling via Graph Neural Network and Deep Reinforcement Learning) model found in the `solution_methods/FJSP_DRL` directory.

## Overview

The model uses **Deep Reinforcement Learning (DRL)**, specifically **Proximal Policy Optimization (PPO)**, combined with a **Heterogeneous Graph Neural Network (HGNN)** to solve the Flexible Job Shop Scheduling Problem (FJSP). It learns a policy to dynamically select the next operation and the machine to process it, aiming to minimize the makespan.

## 1. Model Architecture

The core of the solution is the `HGNNScheduler` (defined in `src/PPO.py`), which uses a graph representation of the scheduling state.

### Heterogeneous Graph Neural Network (HGNN)
The problem is modeled as a heterogeneous graph with two types of nodes:
*   **Operation Nodes**: Represent the operations of jobs.
*   **Machine Nodes**: Represent the available machines.

The graph updates node embeddings through message passing layers based on different relationships:
*   **Machine-Operation Edges**: Connections between machines and the operations they can process. Weighted by processing times.
*   **Precedence Edges**: Connections between consecutive operations of the same job.
*   **Sub-Operation Edges**: (Likely related to alternative processing options or specific constraints, defined in `ope_sub_adj`).

### Node Embeddings (`network/hgnn.py` & `src/PPO.py`)
*   **Machine Embeddings**: Updated using `GATedge` (Graph Attention Network with edge features). It aggregates information from connected operation nodes, considering the processing efficiency (edge features).
*   **Operation Embeddings**: Updated using `MLPs`. It aggregates information from:
    *   Connected Machine nodes.
    *   Preceding Operation nodes (precedence constraints).
    *   Sub-Operation nodes.
    *   Self-embedding.

The embeddings are updated iteratively (controlled by `num_heads` length in config, typically 1 or more layers).

### Actor-Critic Network
The system uses an Actor-Critic architecture for PPO:

*   **Actor (`MLPActor`)**:
    *   **Input**: Concatenation of embeddings for:
        *   Current Job (waiting operation)
        *   Candidate Machine
        *   Pooled Operation features
        *   Pooled Machine features
    *   **Output**: A score for each (Job, Machine) pair.
    *   **Masking**: Invalid actions (e.g., machine cannot process operation, job not ready) are masked out.
    *   **Action**: Selects a pair of (Operation, Machine) to schedule next.

*   **Critic (`MLPCritic`)**:
    *   **Input**: Pooled embeddings of all Operations and Machines.
    *   **Output**: A scalar Value representing the quality of the current state (expected reward).

## 2. Training Process (`train_FJSP_DRL.py`)

The training script trains the PPO agent using reinforcement learning.

*   **Configuration**: Parameters are loaded from `configs/FJSP_DRL.toml`.
*   **Environment**: Uses `FJSPEnv_training`.
*   **Instance Generation**:
    *   Instead of training on a fixed dataset, it generates random FJSP instances using `CaseGenerator` on the fly.
    *   This improves generalization.
*   **Training Loop**:
    1.  **Rollout**: The `policy_old` interacts with the environment to collect trajectories (states, actions, rewards).
    2.  **Storage**: Transitions are stored in `Memory`.
    3.  **Update**: Every `update_timestep`:
        *   Calculates discounted rewards.
        *   Updates the `policy` using the PPO loss function (maximizing advantage while clipping updates).
        *   Weights are copied to `policy_old`.
    4.  **Validation**: Periodically evaluates the model on a fixed validation set (`env_parameters["valid_batch_size"]`).
    5.  **Saving**: Saves the model if the validation makespan improves.

## 3. Running / Inference (`run_FJSP_DRL.py`)

The running script loads a trained model and solves specific FJSP instances.

*   **Parameters**: Uses `test_parameters` from the config file.
*   **Model Loading**: Loads the `.pt` file specified in `trained_policy`.
*   **Modes**:
    *   **Static**: Solves a single benchmark instance (loaded by `load_job_shop_env`).
    *   **Online/Dynamic**: Simulated environment where jobs arrive over time (`online_arrivals=True`).
*   **Execution**:
    *   The `hgnn_model.act` function is called iteratively to select operations until all jobs are finished.
    *   Can use **Greedy** strategy (max probability) or **Sampling** strategy (based on probability) defined by `sample` parameter.
*   **Output**:
    *   **Makespan**: The total time to complete all jobs.
    *   **Gantt Chart**: Visualizes the schedule (if enabled).
    *   **Results**: Saved to `results/` directory.

## Summary of Files

| File | Description |
| :--- | :--- |
| `train_FJSP_DRL.py` | Main script to train the PPO model. |
| `run_FJSP_DRL.py` | Main script to evaluate/run the trained model on instances. |
| `src/PPO.py` | Defines `PPO` agent, `Memory`, `HGNNScheduler` (the full model), and `MLPs` (part of embedding). |
| `network/hgnn.py` | Defines Graph Neural Network layers: `GATedge` (Machine embedding) and `MLPsim` (Operation embedding part). |
| `src/env_training.py` | RL Environment for training phase. |
| `src/env_test.py` | RL Environment for testing/inference phase. |
