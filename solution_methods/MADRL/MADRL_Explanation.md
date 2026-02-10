# MADRL (Multi-Agent Deep Reinforcement Learning) for FJSP

The **MADRL** implementation in this repository is designed to solve the **Flexible Job Shop Scheduling Problem (FJSP)** using a multi-agent reinforcement learning approach. It builds upon the **Dual Attention Network (DAN)** architecture introduced in the **DANIEL** model but reframes the decision-making process into an agent-based paradigm.

---

## 1. Core Concept: Multi-Agent Formulation

In this MADRL formulation:
- **Agents**: Each **Machine** is treated as an autonomous agent.
- **Action Space**: For each machine (agent), the action is to select a **Job** from the pool of currently available jobs to process next.
- **Decision making**: In each decision step, all machines simultaneously (or semi-synchronously) decide which job to take.

### Decision Step Workflow
The environment's `step_madrl` function handles the multi-agent interaction:
1. **Shuffle**: Machines are processed in a random order to ensure fairness and stochasticity.
2. **Action Validation**: For each machine, the selected job is checked for compatibility and availability (the next operation of that job must be executable on that machine).
3. **Sequential Update**: The state is updated incrementally as each machine's choice is processed.

---

## 2. Model Architecture

The model uses the **PPO (Proximal Policy Optimization)** algorithm and consists of a feature extraction backbone and decision heads.

### Feature Extraction: Dual Attention Network (DAN)
The model reuses the `DualAttentionNetwork` from DANIEL to extract high-level representations:
- **Operation-Message Attention (OAB)**: Captures dependencies between operations within the same job and across machines.
- **Machine-Message Attention (MAB)**: Captures the workload and status of machines.

### Actor and Critic
- **Actor**: Takes the concatenated features of Operations, Machines, and **Pairwise compatibility** to output a score matrix of shape `[Jobs, Machines]`.
- **Critic**: Estimates the state value ($V$) based on the global features of operations and machines to guide the PPO training.

---

## 3. State Representation (Features)

The model uses a rich set of features to represent the state of the scheduling environment:

### Operation Features (10-dim)
1. **Scheduled Flag**: Whether the operation is already scheduled.
2. **Completion Time LB**: Lower bound of the completion time.
3. **Min PT**: Minimum processing time across compatible machines.
4. **PT Span**: Difference between max and min processing times.
5. **Mean PT**: average processing time across compatible machines.
6. **Waiting Time**: Time spent waiting to be scheduled.
7. **Job Remain Work**: Sum of processing times for remaining operations in the job.
8. **Left Op Nums**: Number of remaining operations in the job.
9. **Total Remain Work**: Global remaining work in the environment.
10. **Compatible Mch Nums**: Number of machines that can process this operation.

### Machine Features (8-dim)
1. **Available Jobs Count**: Number of available jobs compatible with this machine.
2. **Available Ops Count**: total operations compatible with this machine.
3. **Min PT**: Min processing time for currently available operations.
4. **Mean PT**: Mean processing time for currently available operations.
5. **Waiting Time**: Current idle time.
6. **Remain Work**: Total workload assigned to this machine.
7. **Free Time**: Time when the machine becomes available.
8. **Working Flag**: Binary flag indicating if the machine is currently busy.

### Pair Features (8-dim)
These features capture the specific compatibility between a **Candidate Operation** (next in job) and a **Machine**:
- Processing time of the candidate on the target machine.
- Normalized processing times (relative to job max, machine max, global max).
- Potential wait time if the candidate is assigned to this machine.

---

## 2. Execution and Training

### Reward Design
The model uses a **dense reward** based on the reduction of the makespan lower bound:
$$ Reward_{t} = MakespanLB_{t-1} - MakespanLB_{t} $$
This encourages the agents to make decisions that minimize the overall schedule length at every step.

### Optimization
- **Algorithm**: PPO (Proximal Policy Optimization).
- **Masking**: A dynamic mask is applied to prevent machines from selecting incompatible jobs or jobs that have already completed all their operations.
- **Exploration**: Controlled via entropy regularization and softmax temperature during sampling.

---

## 3. Summary Table

| Feature | Description |
| :--- | :--- |
| **Backbone** | Dual Attention Network (DAN) |
| **Agent Definition** | Machines (Decide which Job to process) |
| **Reward** | Change in Makespan Lower Bound (Dense) |
| **Policy** | Multi-discrete (one Job per Machine) |
| **Framework** | Multi-Agent PPO |
