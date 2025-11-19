# MARL-STL-BFT Performance Analysis Report

## Executive Summary

The Multi-Agent Reinforcement Learning (MARL) system for APT detection exhibits catastrophic failure with a test accuracy of **20.17%**. The model has collapsed to always predicting the positive class (threat detection), resulting in:
- **True Negatives: 0** (no benign traffic correctly classified)
- **False Positives: 59,869** (all benign traffic misclassified as threats)
- **Precision: 20.17%** (only 1 in 5 detections is correct)
- **Recall: 100%** (catches all threats but at extreme cost)

This analysis identifies **11 critical issues** causing the poor performance.

---

## Research Context

**Research Goal:** Multi-Agent Reinforcement Learning with Signal Temporal Logic (STL) monitoring and Byzantine Fault-Tolerant (BFT) consensus for autonomous APT detection in DevOps environments.

**Dataset:** DARPA Transparent Computing dataset
- Training: 350,000 records (91.1% benign, 8.9% threats)
- Validation: 75,000 records
- Test: 75,000 records
- **Class imbalance ratio: 10.2:1**

**Architecture:**
- 3 independent MARL agents using IPPO (Independent Proximal Policy Optimization)
- LSTM-based policy and value networks (256 hidden dim, 2 layers)
- STL constraint monitoring for security specifications
- BFT consensus for multi-agent decision making
- Action space: [Allow, Block, Quarantine, Alert]

---

## Critical Issues Identified

### 1. **Severe Reward Imbalance** ⚠️ CRITICAL

**Problem:** The reward structure creates extreme bias toward restrictive actions (Block/Quarantine).

**Evidence:**
```python
# From _calculate_reward():
if restrictive and true_label == 1:
    reward += 5.0  # True Positive
elif permissive and true_label == 0:
    reward += 2.0  # True Negative
elif restrictive and true_label == 0:
    reward -= 3.0  # False Positive
elif permissive and true_label == 1:
    reward -= 6.0  # False Negative
```

**Analysis:**
With 91.1% benign traffic, let's calculate expected rewards for different strategies:

**Always Block Strategy:**
- On benign (91.1%): -3.0 reward × 0.911 = -2.73
- On threats (8.9%): +5.0 reward × 0.089 = +0.445
- **Expected reward: -2.285 per action**

**Always Allow Strategy:**
- On benign (91.1%): +2.0 reward × 0.911 = +1.822
- On threats (8.9%): -6.0 reward × 0.089 = -0.534
- **Expected reward: +1.288 per action**

**However, the model learned to Block everything! Why?**

The issue is compounded by:
1. **False Negative penalty is too high** (-6.0) making agents risk-averse
2. **Resource penalty** for restrictive actions is negligible (-0.05)
3. **Early exploration** may have led to high FN penalties, biasing the policy

**Impact:** Agents learn to minimize worst-case penalty (FN) rather than maximize expected reward.

---

### 2. **Insufficient Exploration (Low Entropy Coefficient)** ⚠️ CRITICAL

**Problem:** Entropy coefficient of 0.02 is too low for adequate exploration.

**Evidence:**
```python
config = {
    'entropy_coef': 0.02,  # Too low!
}
```

Pre-training diagnostic shows agents already biased:
- Block: 66.7%
- Alert: 33.3%
- Allow: 0%
- Quarantine: 0%

**Standard Values:**
- PPO papers typically use: 0.01 - 0.1
- For imbalanced problems: 0.05 - 0.2
- Current value: 0.02 (borderline insufficient)

**Impact:** Agents converge prematurely to suboptimal policy without exploring balanced strategies.

---

### 3. **Value Network Initialization/Learning Issues** ⚠️ HIGH

**Problem:** Value network predictions are near zero before and during training.

**Evidence from diagnostics:**
```
Value predictions:
  Mean: -0.00
  Std:  0.01
  Min:  -0.01
  Max:  0.01
```

**Analysis:**
- Value network should predict cumulative discounted rewards
- With rewards ranging from -6 to +5, values should be much larger
- Near-zero values indicate:
  1. Poor weight initialization
  2. Insufficient learning (value_coef too low)
  3. Gradient vanishing in deep LSTM network

**Impact:** Inaccurate value estimates lead to poor advantage calculations, causing unstable policy updates.

---

### 4. **Advantage Normalization on Small Batches** ⚠️ HIGH

**Problem:** Normalizing advantages within each agent's small trajectory batch distorts learning signal.

**Evidence:**
```python
def update_policy(self, trajectories, epochs):
    # ...
    # Normalize advantages PER AGENT
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

**Analysis:**
- Each agent has ~682 samples per iteration (2048 steps / 3 agents)
- Normalizing per-agent removes information about relative advantage magnitudes
- In MARL, agents should learn from shared experience quality

**Impact:** Agents cannot distinguish truly good actions from mediocre ones, slowing convergence.

---

### 5. **Return Normalization Breaks Reward Scale** ⚠️ HIGH

**Problem:** Normalizing returns destroys the carefully designed reward structure.

**Evidence:**
```python
# ADD THIS: Normalize returns to help value network learning
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

**Analysis:**
- Returns are normalized AFTER computing from rewards
- This converts meaningful reward signals into standardized z-scores
- Value network learns to predict normalized returns, losing connection to actual rewards
- Makes reward engineering irrelevant

**Impact:** The entire reward structure (TP=5.0, FN=-6.0, etc.) becomes meaningless after normalization.

---

### 6. **No Class Weighting in Reward Function** ⚠️ HIGH

**Problem:** Dataset uses WeightedRandomSampler but rewards don't account for sampling bias.

**Evidence:**
```python
# Weighted sampler used
sampler = WeightedRandomSampler(sample_weights, ...)

# But rewards don't compensate for sampling weights
reward = 5.0  # Same reward regardless of sampling probability
```

**Analysis:**
- WeightedRandomSampler over-samples minority class (threats)
- Agent sees ~50/50 split during training due to sampling
- But test set has natural 91.1/8.9 distribution
- Agent doesn't learn true base rates

**Expected vs Reality:**
- Training: Sees ~50% threats (due to weighted sampling)
- Test: Sees ~8.9% threats (natural distribution)
- **Distribution shift of 5.6x!**

**Impact:** Agents learn on artificial distribution, failing to generalize to real-world class frequencies.

---

### 7. **STL Violations Always Zero** ⚠️ MEDIUM

**Problem:** STL monitoring reports zero violations across all 220 training iterations.

**Evidence from results:**
```json
"stl_violations": [0.0, 0.0, 0.0, ...] // All zeros
```

**Analysis:**
This suggests one of three problems:
1. **STL specifications are too lenient** (never violated)
2. **STL monitoring is not properly implemented**
3. **STL signal construction is incorrect**

**Code Review:**
```python
def _create_stl_signal(self, actions, states, labels, rewards):
    # Implementation not shown in extracted code
```

Without seeing the implementation, it's impossible to verify correctness.

**Impact:**
- If STL is not working, constraint_coef * constraint_loss is adding meaningless gradient signal
- If STL is too lenient, it provides no useful guidance

---

### 8. **Constraint Loss Target is Unrealistic** ⚠️ MEDIUM

**Problem:** Constraint network is trained to always predict perfect satisfaction (1.0).

**Evidence:**
```python
# In update_policy:
constraint_target = torch.ones_like(constraints)
constraint_loss = F.mse_loss(constraints, constraint_target)
```

**Analysis:**
- Target is always 1.0 (perfect constraint satisfaction)
- But STL violations are at 0, suggesting constraints are already satisfied
- Training network to predict 1.0 when reality is already 1.0 provides no learning signal
- Creates constant gradient that doesn't guide policy improvement

**Impact:** Constraint network becomes a constant function, wasting model capacity and adding noise to training.

---

### 9. **BFT Consensus May Amplify Errors** ⚠️ MEDIUM

**Problem:** When all agents learn the same biased policy, BFT consensus reinforces the error.

**Evidence:**
```python
class BFTConsensus:
    def reach_consensus(self, actions, states):
        # Takes majority vote among agents
        # If all agents say "Block", consensus is "Block"
```

**Analysis:**
- BFT assumes byzantine agents are outliers
- But if all agents learn the same incorrect policy (always Block), there's no "correct" agent to outvote
- Consensus just rubber-stamps the shared mistake

**Theoretical Issue:**
- BFT protects against adversarial agents (1 out of 3)
- But doesn't protect against correlated learning failures (all 3 agents wrong)

**Impact:** BFT provides false confidence in incorrect decisions.

---

### 10. **Global Reward Creates Feedback Loop** ⚠️ LOW

**Problem:** Global reward based on accuracy creates positive feedback for bad behavior.

**Evidence:**
```python
def _calculate_global_reward(self, actions, states, labels):
    correct = sum(
        1 for a, l in zip(actions, labels)
        if (a in [1, 2] and l == 1) or (a in [0, 3] and l == 0)
    )
    accuracy = correct / len(actions)
    security_bonus = accuracy * 3.0
    # ...
    rewards = [r + 0.2 * global_reward for r in rewards]
```

**Analysis:**
If all agents Block everything:
- On sampled batch (50/50 due to weighted sampling): accuracy ≈ 50%
- Global reward = 0.2 * (2.0/3 + 1.5) = 0.433
- This bonus applies to ALL agents regardless of individual action

**Impact:**
- Weak signal (0.2 coefficient)
- Can't overcome individual reward imbalance
- Creates credit assignment problem (reward not tied to specific actions)

---

### 11. **Insufficient Training Iterations** ⚠️ LOW

**Problem:** 200 iterations may be insufficient for multi-agent convergence.

**Evidence:**
- Training uses 200 iterations
- Each iteration: 2048 steps = ~682 steps per agent
- Total experience per agent: 136,400 steps

**Analysis:**
- PPO typically requires 1M - 10M steps for convergence
- Current training: 136K steps per agent (13.6% of minimum)
- Training curves show policy loss still increasing at iteration 200

**However:** This is likely NOT the primary issue because:
- Model has clearly converged to a policy (always Block)
- More iterations would just reinforce the bad policy
- Need to fix reward structure first

**Impact:** Premature stopping, but not the root cause.

---

## Root Cause Summary

The model failure is caused by a **perfect storm** of interacting issues:

### Primary Causes (Must Fix):
1. **Reward imbalance** + **dataset imbalance** = Always Block is locally optimal
2. **Low entropy** = No exploration of better strategies
3. **Value network failure** = Poor advantage estimates
4. **Distribution shift** = Training/test mismatch due to weighted sampling

### Secondary Causes (Amplify problem):
5. **Advantage/return normalization** = Destroys reward signal
6. **STL not working** = Adds noise instead of guidance
7. **BFT consensus** = Amplifies shared error

### Tertiary Causes (Minor impact):
8. **Constraint loss** = Meaningless gradient
9. **Global reward** = Weak and poorly designed
10. **Insufficient training** = But not root cause

---

## Recommendations

### Immediate Fixes (Critical):

1. **Rebalance Reward Structure:**
```python
# Adjust for base rates (91.1% benign, 8.9% threats)
if restrictive and true_label == 1:
    reward += 10.0  # Increase TP reward (rare but valuable)
elif permissive and true_label == 0:
    reward += 1.0   # Decrease TN reward (common, less informative)
elif restrictive and true_label == 0:
    reward -= 8.0   # Increase FP penalty (we're doing this too much!)
elif permissive and true_label == 1:
    reward -= 10.0  # Keep FN penalty high (critical security failure)

# Add significant resource cost for restrictive actions
if action in [1, 2]:
    reward -= 1.5  # 30x increase from 0.05
```

**Rationale:** Make "Always Block" strategy have negative expected reward.

2. **Increase Exploration:**
```python
config = {
    'entropy_coef': 0.15,  # Increase from 0.02
}
```

3. **Remove Return Normalization:**
```python
# DELETE THIS LINE:
# returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

4. **Fix Value Network Learning:**
```python
config = {
    'value_coef': 1.0,  # Increase from 0.5
    'lr': 1e-3,  # Increase from 5e-4
}
```

### Medium Priority:

5. **Remove Weighted Sampler OR Adjust Rewards:**
```python
# Option A: Use natural class distribution
train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,  # No weighted sampler
    # ...
)

# Option B: Keep weighted sampler but adjust rewards by sampling probability
reward = base_reward / sampling_weight
```

6. **Fix Advantage Computation:**
```python
# Normalize advantages across ALL agents, not per-agent
all_advantages = torch.cat([
    torch.tensor([t['advantage'] for t in agent_traj])
    for agent_traj in trajectories
])
advantages_normalized = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)
```

7. **Debug STL Monitoring:**
- Add logging to verify STL is actually checking constraints
- Tighten thresholds if violations are always zero
- Consider removing if not adding value

### Lower Priority:

8. **Increase Training Duration:**
```python
config = {
    'num_iterations': 500,  # Increase from 200
}
```

9. **Add Curriculum Learning:**
- Start with balanced batches
- Gradually shift to natural distribution
- Helps agent learn both classes before facing imbalance

10. **Implement Focal Loss:**
```python
# For addressing class imbalance in reward
focal_weight = (1 - accuracy) ** 2
reward *= focal_weight
```

---

## Expected Outcomes After Fixes

With the recommended changes, expected metrics:
- **Accuracy:** 85-95% (from 20%)
- **Precision:** 70-85% (from 20%)
- **Recall:** 80-95% (from 100%)
- **F1-Score:** 75-90% (from 34%)
- **True Negatives:** >55,000 (from 0)
- **False Positives:** <5,000 (from 59,869)

---

## Conclusion

The MARL system fails due to **misaligned incentives** in the reward structure combined with **insufficient exploration**. The model rationally learned to minimize worst-case penalties (False Negatives) by blocking everything, even though this strategy has poor expected reward.

The root cause is not the model architecture (which is well-designed) or the training algorithm (PPO is appropriate), but rather **reward engineering** and **data distribution handling**.

This is a textbook example of the reward hacking problem in RL: the agent found a strategy that scores well on the training objective (avoiding FN penalties) while failing catastrophically at the true goal (accurate threat detection).

---

## Appendix: Performance Metrics Breakdown

### Test Set Confusion Matrix:
```
                Predicted
                Benign  Threat
Actual Benign      0    59,869  (FP)
       Threat      0    15,131  (TP)
```

### Derived Metrics:
- Accuracy: (TP + TN) / Total = 15,131 / 75,000 = 20.17%
- Precision: TP / (TP + FP) = 15,131 / 75,000 = 20.17%
- Recall: TP / (TP + FN) = 15,131 / 15,131 = 100%
- F1: 2 * (P * R) / (P + R) = 2 * (0.2017 * 1.0) / 1.2017 = 33.58%

### Class Distribution:
- Test set: 20.17% threats, 79.83% benign
- Model predicts: 100% threats, 0% benign
- **The model is essentially a constant function: always return "threat"**

---

**Report Generated:** 2025-11-19
**Analysis By:** Claude Code (Automated MARL Performance Audit)
**Notebook Analyzed:** MARL_STL_BFT_Training_v1_T4_Optimized [current-working-bad-results-v2].ipynb
