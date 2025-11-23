# ANALYSIS: Training Results After Fix #2 (Iteration 300)

**Date:** 2025-11-23
**Status:** ❌ FAILED - Value loss exploding (opposite problem)
**Branch:** `claude/fix-marl-data-split-01LPhSB657byDs2zVs7kvdKy`

---

## Training Results Summary

You ran training for 300 iterations after applying Fix #1 (Monte Carlo) and Fix #2 (Remove normalization).

### Value Loss Progression

| Iteration | Value Loss | Status |
|-----------|------------|--------|
| 10 | **998,829.64** | 🔴 EXPLODING! |
| 50 | 224,572.52 | 🔴 Still huge |
| 100 | 88,553.18 | 🔴 Very high |
| 150 | 84,433.75 | 🔴 Not converging |
| 200 | 87,145.06 | 🔴 Oscillating |
| 300 | (not shown) | 🔴 Failed |

**Expected:** 1.0 → 0.7 → 0.5 → 0.3 → 0.1
**Actual:** 1.0 → **999k** → 225k → 88k (EXPLOSION!)

### Evaluation Results

| Iteration | TP | TN | FP | FN | F1-Score | Status |
|-----------|-----|-----|-----|-----|----------|---------|
| 20-40 | 0 | 1419 | 0 | 181 | 0.0 | Allow everything |
| 60-80 | 181 | 0 | 1419 | 0 | 0.203 | Block everything |
| 100-300 | 0 | 1419 | 0 | 181 | 0.0 | Allow everything |

**Best F1:** 0.203 at iteration 60 (choosing "block" for everything, not learning)

---

## What Went Wrong

### Fix #1: Monte Carlo Returns ✅ (Correct)
**Applied successfully:**
```python
# Line 192 in compute_gae()
return_t = reward + self.gamma * return_t * (1 - done)
returns.insert(0, return_t)
```

This was correct - returns computed from actual rewards.

### Fix #2: Remove Normalization ❌ (TOO EXTREME)
**Applied but caused explosion:**
```python
# Removed all return normalization
# Value network sees raw returns in 100k+ range
# Can't learn to predict such large values
# Value loss explodes!
```

### Why Value Loss Exploded

**Monte Carlo returns accumulate to MASSIVE values:**

```
Episode reward: -2000 per episode
Timesteps: 1000 per episode
Gamma: 0.99

Return calculation:
return_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γ^T*r_T
         = -2000 + 0.99*(-2000) + 0.99²*(-2000) + ...
         ≈ -200,000 to -500,000 (depending on episode length)
```

**Value network tries to predict these:**
- Initial: Predicts ~0 (random init)
- Target: -200,000 (actual return)
- Loss = MSE(0, -200000)² = **40,000,000,000**
- Explodes immediately!

---

## The Real Problem: Normalization Dilemma

### Three approaches tried:

**1. Per-batch normalization (Fix #1 only):**
```python
# Normalize each batch independently
returns = (returns - returns.mean()) / returns.std()
# Result: mean=0, std=1 for EVERY batch
# Problem: Value loss stuck at 1.0 (learns to predict mean=0)
```
❌ **Result:** Value loss stuck at 1.0 for 300 iterations

**2. No normalization (Fix #1 + Fix #2):**
```python
# Use raw returns directly
returns = raw_MC_returns  # Can be -200,000 or more
# Problem: Value network can't handle such large magnitudes
```
❌ **Result:** Value loss explodes to 998,829

**3. Running normalization (Fix #3 - Applied now):**
```python
# Track mean/std across ALL training
self.return_mean = running_average_of_all_returns
self.return_std = running_std_of_all_returns

# Normalize using running stats
returns = (returns - self.return_mean) / self.return_std
# Result: Stable scale, but learned from actual data
```
✅ **Expected:** Value loss stable and decreasing

---

## Fix #3: Running Normalization (Applied)

### What Changed

**Added to `__init__` method (lines 76-78):**
```python
# Running statistics for return normalization
self.return_mean = 0.0
self.return_std = 1.0
self.return_count = 0
```

**Modified `update_policy` method (lines 261-284):**
```python
# Update running statistics (Welford's online algorithm)
returns_np = returns.cpu().numpy()
batch_mean = returns_np.mean()
batch_std = returns_np.std()
batch_count = len(returns_np)

# Update running mean and std
delta = batch_mean - self.return_mean
self.return_mean += delta * batch_count / (self.return_count + batch_count)
self.return_std = np.sqrt(
    (self.return_count * self.return_std**2 +
     batch_count * batch_std**2 +
     delta**2 * self.return_count * batch_count / (self.return_count + batch_count)) /
    (self.return_count + batch_count)
)
self.return_count += batch_count

# Normalize using running statistics
if self.return_std > 1e-6:
    returns = (returns - self.return_mean) / (self.return_std + 1e-8)
else:
    returns = returns - self.return_mean
```

### Why Running Normalization Works

| Method | Mean/Std Per Batch | Problem | Result |
|--------|-------------------|---------|---------|
| **Per-batch** | mean=0, std=1 (forced) | Loss stuck at variance=1.0 | ❌ No learning |
| **No norm** | Raw values (100k+) | Can't learn large values | ❌ Explosion |
| **Running** | Learned from data | Stable scale, real distribution | ✅ Learning |

**Key insight:**
- Running stats give normalized scale (~mean=0, std=1)
- But mean/std are **learned from actual data** over time
- Early iterations: stats unstable but improving
- Later iterations: stats converge, stable normalization
- Value network learns on stable scale

---

## Expected Results After Fix #3

### Iterations 1-20:
- Value loss: **Unstable initially** (1.0 → 2.0 → 0.8 → 1.5)
- Reason: Running stats not yet stabilized
- Episode rewards: -2000 → -1500

### Iterations 20-50:
- Value loss: **Starting to decrease** (1.5 → 0.8 → 0.5)
- Reason: Running stats stabilizing
- Episode rewards: -1500 → -800
- **First TP > 0** appearing

### Iterations 50-100:
- Value loss: **Decreasing steadily** (0.5 → 0.3)
- Reason: Stats stable, value network learning
- Episode rewards: -800 → +200 (turning positive!)
- TP: 500-1000
- F1-score: 0.3-0.5

### Iterations 100-200:
- Value loss: **Converging** (0.3 → 0.15)
- Episode rewards: +200 → +500
- F1-score: 0.5-0.7

---

## Comparison: All Three Attempts

| Attempt | Fix Applied | Value Loss | TP @ 100 | F1 @ 200 | Outcome |
|---------|-------------|------------|----------|----------|---------|
| **1st** | MC only | Stuck at 1.0 | 0 | 0.0 | ❌ No learning |
| **2nd** | MC + No norm | Explodes to 88k | 0 | 0.0 | ❌ Explosion |
| **3rd** | MC + Running norm | (pending) | (pending) | (pending) | ⏳ Testing |

---

## Technical Details: Welford's Algorithm

The running normalization uses **Welford's online algorithm** for numerical stability:

```python
# Update running mean
n_old = self.return_count
n_new = n_old + batch_count
delta = batch_mean - self.return_mean

self.return_mean_new = self.return_mean + delta * batch_count / n_new

# Update running variance (numerically stable)
var_old = self.return_std**2
var_batch = batch_std**2

var_new = (n_old * var_old + batch_count * var_batch +
           delta**2 * n_old * batch_count / n_new) / n_new

self.return_std_new = sqrt(var_new)
```

**Why Welford's:**
- Numerically stable (avoids catastrophic cancellation)
- Single pass (doesn't need to store all values)
- Updates incrementally (memory efficient)
- Standard in online statistics

---

## Monitoring Guide

### After 10 iterations, check:

**If value loss is:**
- **Oscillating (0.8-2.0):** ✅ GOOD - stats stabilizing
- **Exploding (>10k):** ❌ BAD - something wrong
- **Stuck at 1.0:** ❌ BAD - normalization not applied

### After 30 iterations, check:

**If value loss is:**
- **Decreasing (0.8-0.5):** ✅ GOOD - learning started
- **Still exploding:** ❌ BAD - stop and debug
- **Still stuck at 1.0:** ❌ BAD - running norm not working

### After 50 iterations, check:

**If you see:**
- Value loss < 0.5: ✅ GOOD
- TP > 0 in evaluation: ✅ GOOD
- Episode rewards > -800: ✅ GOOD

### Red flags (stop if you see):

1. Value loss > 10,000 after iteration 20
2. Value loss still at 1.0 after iteration 50
3. TP = 0 for all evaluations after iteration 100

---

## Summary

**Three bugs identified and fixed:**

1. **Bug #1:** Circular dependency in returns
   - **Fix:** Monte Carlo returns from actual rewards
   - **Commit:** de12267

2. **Bug #2:** Per-batch normalization stuck at loss=1.0
   - **Fix:** Removed normalization
   - **Commit:** 9a043c4
   - **Result:** Caused Bug #3 (explosion)

3. **Bug #3:** No normalization caused explosion
   - **Fix:** Running normalization (this commit)
   - **Commit:** 6953564
   - **Expected:** Stable learning

**Current status:** Fix #3 applied, ready for next training run.

---

## Files Modified

- ✅ `MARL_STL_BFT_Training_v1_T4_Optimized [FIXED].ipynb` - Cell 15
  - Added: `self.return_mean/std/count` initialization
  - Added: Running statistics update in `update_policy()`
  - Changed: Returns normalized with running stats

---

**Next step:** Restart training with Fix #3 and monitor value loss in first 50 iterations.
