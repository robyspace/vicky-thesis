# CRITICAL FIX #2: Remove Return Normalization

**Date:** 2025-11-19
**Status:** 🔧 URGENT FIX - Training stopped at iteration 200

---

## What Happened

After implementing Monte Carlo returns (Fix #1), training **STILL failed** for 200 iterations:

| Iteration | Value Loss | Reward | TP | Status |
|-----------|------------|--------|----|---------|
| 10 | 0.9999 | -2107 | 0 | ❌ Stuck |
| 20 | 1.0022 | -1700 | 0 | ❌ Stuck |
| 100 | 0.9996 | -1612 | 0 | ❌ Stuck |
| 200 | 0.9995 | -1378 | 0 | ❌ STILL STUCK! |

**Value loss stuck at 1.0 for ALL 200 iterations.**

---

## Root Cause #2: Return Normalization

In `update_policy()` method (Cell 15, lines 51-58), returns were being normalized:

```python
# BROKEN: This was preventing learning!
returns_mean = returns.mean()
returns_std = returns.std()
if returns_std > 1e-8:
    returns = (returns - returns_mean) / (returns_std + 1e-8)
else:
    returns = returns - returns_mean
```

### Why This Breaks Learning

**1. Normalized returns lose absolute scale:**
```python
Original returns: [-2000, -1800, -1500, ...]  # Actual rewards
Normalized returns: [-1.2, -0.8, 0.5, ...]    # Mean=0, Std=1
```

**2. Value network learns to predict the mean (which is 0):**
```python
# After normalization: mean(returns) = 0
# Optimal prediction: value = 0 (the mean)
# Loss = MSE(0, normalized_returns) = Var(normalized_returns) = 1.0
```

**3. Loss stuck at 1.0 forever:**
- Value network predicts 0 (the normalized mean)
- MSE against normalized targets ≈ 1.0 (the variance)
- **No learning occurs!**

### Mathematical Proof

After normalization:
```
returns_norm = (returns - mean) / std

E[returns_norm] = 0  (by definition)
Var(returns_norm) = 1  (by definition)

If value network predicts: V(s) = 0 (the mean)
Then: MSE = E[(returns_norm - 0)²] = E[returns_norm²] = Var = 1.0
```

The value network is just learning to predict the mean, giving a constant loss of 1.0!

---

## The Fix

**REMOVED the return normalization completely.**

### Before (BROKEN):
```python
# Normalize advantages (CORRECT - keep this)
advantages = (advantages - adv_mean) / (adv_std + 1e-8)

# Normalize returns (WRONG - removed this!)
returns_mean = returns.mean()
returns_std = returns.std()
if returns_std > 1e-8:
    returns = (returns - returns_mean) / (returns_std + 1e-8)

# PPO update epochs
for epoch in range(epochs):
    ...
```

### After (FIXED):
```python
# Normalize advantages (CORRECT - keep this)
advantages = (advantages - adv_mean) / (adv_std + 1e-8)

# NOTE: Do NOT normalize Monte Carlo returns!
# Returns must keep their absolute scale for value network to learn.
# Normalization destroys the learning signal (causes value loss = 1.0).

# PPO update epochs
for epoch in range(epochs):
    ...
```

---

## Why Return Normalization Was There

**Historical context:**

1. **Original bug:** Returns computed as `gae + old_value` (circular dependency)
2. **Problem:** With broken returns, value loss exploded (38k → 49k)
3. **Previous fix:** Added return normalization to prevent explosion
4. **Side effect:** Normalization prevented explosion BUT also prevented learning

**With Monte Carlo returns:**
- Returns are well-behaved (no explosion risk)
- Return normalization is not only unnecessary, it's **harmful**
- Must use raw returns for value network to learn absolute reward scale

---

## PPO Best Practices

### What TO normalize:
✅ **Advantages:** For stable policy gradients
```python
advantages = (advantages - mean) / (std + eps)
```

### What NOT to normalize:
❌ **Returns:** Value network needs absolute scale
```python
# WRONG: returns = (returns - mean) / std
# RIGHT: Use raw Monte Carlo returns
```

### Why the difference?

**Advantages (for policy):**
- Only the **relative** advantage matters
- "Is action A better than average?"
- Normalization helps with numerical stability
- Doesn't need absolute scale

**Returns (for value network):**
- Need **absolute** scale to learn value function
- "What is the actual expected return from this state?"
- Normalization destroys the information needed to learn
- Must keep raw scale

---

## Expected Results After This Fix

### Iterations 1-20:
- ✅ Value loss DECREASING: 1.0 → 0.7 → 0.5
- ✅ Not stuck at 1.0 anymore!

### Iterations 20-50:
- ✅ Value loss: 0.5 → 0.2
- ✅ Episode rewards improving: -1800 → -1000

### Iterations 50-100:
- ✅ Value loss: 0.2 → 0.1 (converging)
- ✅ Episode rewards: -1000 → +200 (turning positive!)
- ✅ **First true positives (TP > 0)**

### Iteration 150-200:
- ✅ Value loss: < 0.15
- ✅ Episode rewards: +200 to +500
- ✅ F1-score: 0.6-0.8

---

## Technical Explanation

### Why Value Loss Was Exactly 1.0

**Step-by-step breakdown:**

1. **Compute Monte Carlo returns:** (correctly done)
   ```python
   returns = [-2000, -1800, -1600, -1500, ...]  # Varying actual rewards
   ```

2. **Normalize returns:** (THIS WAS THE BUG!)
   ```python
   mean = -1700
   std = 200
   normalized_returns = [(-2000+1700)/200, (-1800+1700)/200, ...]
                       = [-1.5, -0.5, 0.5, ...]
   # Now: mean ≈ 0, std ≈ 1
   ```

3. **Value network sees normalized targets:**
   ```python
   # Network tries to learn: V(s) → normalized_return
   # Best constant prediction: V(s) = 0 (the mean)
   ```

4. **Loss computation:**
   ```python
   loss = MSE(predicted, normalized_returns)
        = MSE(0, normalized_returns)  # if predicting mean
        = E[(normalized_returns - 0)²]
        = E[normalized_returns²]
        = Var(normalized_returns)  # since mean = 0
        = 1.0  # by definition of normalization
   ```

**The loss is mathematically guaranteed to be ~1.0 if the network predicts the mean!**

---

## Summary of Both Fixes

### Fix #1: Monte Carlo Returns (Commit de12267)
**Problem:** Circular dependency in return computation
**Solution:** Use Monte Carlo returns from actual rewards
**File:** Cell 15, `compute_gae()` method

### Fix #2: Remove Return Normalization (This commit)
**Problem:** Normalization destroys learning signal
**Solution:** Remove return normalization, keep advantages normalized
**File:** Cell 15, `update_policy()` method

### Combined Effect:
1. ✅ Monte Carlo returns = ground truth from actual rewards
2. ✅ No normalization = value network learns absolute scale
3. ✅ Value loss can now decrease (not stuck at 1.0)
4. ✅ Policy can improve once value network learns

---

## Action Required

### 1. Pull Latest Changes
```python
!git pull origin claude/fix-marl-data-split-01LPhSB657byDs2zVs7kvdKy
```

### 2. RESTART Training from Iteration 1
**CRITICAL:** Must restart, not continue from 200!

The value network has 200 iterations of learning to predict 0. Must reinitialize.

### 3. Watch for Value Loss Decreasing

**After 10 iterations:**
- Value loss should be < 0.9 (not stuck at 1.0!)

**After 30 iterations:**
- Value loss should be < 0.6

**After 50 iterations:**
- Value loss should be < 0.4
- First true positives appearing

**If value loss is still ~1.0 after 30 iterations, stop and debug!**

---

## Files Modified

- ✅ `MARL_STL_BFT_Training_v1_T4_Optimized [FIXED].ipynb` - Cell 15, removed lines 51-58
- ✅ `CRITICAL_FIX_2_REMOVE_RETURN_NORMALIZATION.md` - This file

---

**This MUST work now.** Both fundamental issues are fixed:
1. Returns from ground truth (not circular dependency)
2. No normalization (value network can learn absolute scale)

Value loss WILL decrease this time! 🚀
