# Data Split Issue - Critical Fix Required

**Date:** 2025-11-19
**Status:** BLOCKING - Training cannot succeed with current data split
**Priority:** CRITICAL

---

## Problem Summary

Training is failing because of a **severe distribution mismatch** between training and validation/test sets:

| Split | Total Samples | Threats | Benign | Threat % | Ratio |
|-------|--------------|---------|--------|----------|-------|
| **Training** | 350,000 | 31,249 | 318,751 | **8.9%** | 10.2:1 |
| **Validation** | 75,000 | 780 | 74,220 | **1.04%** | 95.2:1 ⚠️ |
| **Test** | 75,000 | ??? | ??? | ~1% likely | ~95:1 likely ⚠️ |

**The validation set is 10x more imbalanced than training!**

---

## Why This Breaks Training

### Mathematical Proof

On validation set (1% threats), "Always Allow" strategy:
- **Correct TNs:** +1.0 × 1583 = **+1583**
- **Wrong FNs:** -10.0 × 17 = **-170**
- **Net reward:** **+1413** (highly positive!)

This is why evaluation shows:
```
TP=0, TN=1600, FP=0, FN=0
Accuracy=100%, Precision=0, Recall=0, F1=0
```

The model learned that on validation distribution, **not detecting threats is optimal**.

### Training Results Explained

With entropy_coef=0.15, model explores randomly:
- **Training:** 25% each action (still exploring, getting -1600 to -2100 rewards)
- **Evaluation:** 100% permissive actions (learned optimal strategy for 1% threat rate)

The model is stuck because:
1. Training data says "learn to detect threats"
2. Validation data says "don't bother, just allow everything"
3. These objectives conflict → no convergence

---

## Root Cause: Non-Stratified Data Split

The data pipeline (`DARPA_Dataset_Execution_Pipeline.ipynb`) likely splits data **randomly** without stratification:

```python
# WRONG - Current approach (assumed):
train = data[:350000]
val = data[350000:425000]
test = data[425000:]
```

This can create wildly different class distributions if threats are not uniformly distributed in the dataset.

**OR** if using train_test_split without stratify:
```python
# WRONG - Random split without stratification:
train, temp = train_test_split(data, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.333, random_state=42)
# Missing: stratify parameter!
```

---

## Required Fix: Stratified Splitting

### What Needs to Happen

1. **Analyze** `DARPA_Dataset_Execution_Pipeline.ipynb` to find data splitting code
2. **Modify** to use stratified splitting
3. **Re-generate** train/val/test parquet files
4. **Verify** all splits have ~8.9% threat rate
5. **Re-train** model with proper data

### Expected Code Change

**Find this pattern in the notebook:**
```python
# Some version of splitting the data
train_data = ...
val_data = ...
test_data = ...
```

**Replace with stratified splitting:**
```python
from sklearn.model_selection import train_test_split

# Assuming you have full_data and full_labels
# First split: 70% train, 30% temp
train_data, temp_data, train_labels, temp_labels = train_test_split(
    full_data,
    full_labels,
    test_size=0.3,
    stratify=full_labels,  # ← CRITICAL: Ensures same ratio
    random_state=42
)

# Second split: 20% val, 10% test (from the 30% temp)
val_data, test_data, val_labels, test_labels = train_test_split(
    temp_data,
    temp_labels,
    test_size=0.333,  # 0.333 × 0.3 = 0.1 of total
    stratify=temp_labels,  # ← CRITICAL: Ensures same ratio
    random_state=42
)

# Verify splits
print(f"Train: {sum(train_labels)}/{len(train_labels)} = {sum(train_labels)/len(train_labels)*100:.2f}% threats")
print(f"Val:   {sum(val_labels)}/{len(val_labels)} = {sum(val_labels)/len(val_labels)*100:.2f}% threats")
print(f"Test:  {sum(test_labels)}/{len(test_labels)} = {sum(test_labels)/len(test_labels)*100:.2f}% threats")
```

**All three should show ~8.9% threats.**

---

## Alternative: If Data Is Sequential/Temporal

If the DARPA dataset has temporal ordering that must be preserved:

```python
# For temporal data, use time-based split with stratification within time windows
# Split by time first
time_cutoff_1 = data['timestamp'].quantile(0.7)
time_cutoff_2 = data['timestamp'].quantile(0.9)

train_data = data[data['timestamp'] < time_cutoff_1]
val_data = data[(data['timestamp'] >= time_cutoff_1) & (data['timestamp'] < time_cutoff_2)]
test_data = data[data['timestamp'] >= time_cutoff_2]

# Then verify and potentially resample to balance if needed
print(f"Train threats: {train_data['is_suspicious'].mean()*100:.2f}%")
print(f"Val threats:   {val_data['is_suspicious'].mean()*100:.2f}%")
print(f"Test threats:  {test_data['is_suspicious'].mean()*100:.2f}%")

# If severely imbalanced, consider:
# 1. Oversampling threats in val/test, OR
# 2. Using a different time range, OR
# 3. Stratified sampling within time windows
```

---

## After Fix: Update Training Config

Once data is properly split, update hyperparameters in training notebook:

### Cell 22 Config Changes:
```python
config = {
    # ... existing config ...

    # UPDATED: More conservative after data fix
    'lr': 3e-4,           # Reduced from 1e-3
    'entropy_coef': 0.05, # Reduced from 0.15 for better exploitation
    'value_coef': 1.5,    # Increased from 1.0 for stronger value learning
}
```

### Why These Changes:

1. **entropy_coef: 0.15 → 0.05**
   - 0.15 was causing excessive random exploration
   - With proper data, 0.05 allows learning while maintaining diversity

2. **lr: 1e-3 → 3e-4**
   - Slower, more stable learning
   - Prevents oscillation in policy updates

3. **value_coef: 1.0 → 1.5**
   - Stronger value network learning
   - Better advantage estimates for policy gradient

---

## Expected Results After Fix

### With Properly Stratified Data (8.9% threats in all splits):

#### After 30-50 iterations:
- Episode rewards: -1000 to -500 (improving from -2000)
- Action distribution: 60-70% permissive, 30-40% restrictive
- Evaluation: TPs appearing (not all TNs)

#### After 100-150 iterations:
- Episode rewards: **+200 to +500** (POSITIVE!)
- Action distribution: 80-85% permissive, 15-20% restrictive
- Evaluation metrics:
  - Accuracy: **75-85%**
  - Precision: **60-75%**
  - Recall: **70-85%**
  - F1-Score: **65-80%**
  - **TP: ~800-1200** (not 0!)
  - **TN: ~55,000-60,000**
  - **FP: <5,000**

---

## Verification Checklist

Before starting training with new data:

### 1. Data Split Verification
- [ ] Training threats: ~8.9% (30,000-32,000 threats)
- [ ] Validation threats: ~8.9% (6,600-6,800 threats)
- [ ] Test threats: ~8.9% (6,600-6,800 threats)
- [ ] Total samples: 500,000 (350k train, 75k val, 75k test)

### 2. File Generation
- [ ] `train.parquet` created with proper distribution
- [ ] `val.parquet` created with proper distribution
- [ ] `test.parquet` created with proper distribution
- [ ] Files saved to `/content/drive/MyDrive/mythesis/vicky/darpa_tc/splits/`

### 3. Config Update
- [ ] `entropy_coef = 0.05`
- [ ] `lr = 3e-4`
- [ ] `value_coef = 1.5`

### 4. Sanity Check
```python
# Run this after loading data in training notebook:
train_labels = train_dataset.df['is_suspicious'].values
val_labels = val_dataset.df['is_suspicious'].values
test_labels = test_dataset.df['is_suspicious'].values

print("VERIFICATION:")
print(f"Train: {sum(train_labels)}/{len(train_labels)} = {sum(train_labels)/len(train_labels)*100:.2f}%")
print(f"Val:   {sum(val_labels)}/{len(val_labels)} = {sum(val_labels)/len(val_labels)*100:.2f}%")
print(f"Test:  {sum(test_labels)}/{len(test_labels)} = {sum(test_labels)/len(test_labels)*100:.2f}%")

# Should all show ~8.9%
assert 8.0 < sum(train_labels)/len(train_labels)*100 < 10.0, "Train split incorrect!"
assert 8.0 < sum(val_labels)/len(val_labels)*100 < 10.0, "Val split incorrect!"
assert 8.0 < sum(test_labels)/len(test_labels)*100 < 10.0, "Test split incorrect!"
print("✓ All splits properly stratified!")
```

---

## Files to Work With

### In New Thread:

1. **`DARPA_Dataset_Execution_Pipeline.ipynb`**
   - Locate data splitting code
   - Add stratification
   - Regenerate train/val/test parquet files
   - Verify distributions

2. **`MARL_STL_BFT_Training_v1_T4_Optimized [FIXED].ipynb`**
   - Update config (entropy, lr, value_coef)
   - Load new data files
   - Run training
   - Monitor for positive rewards and improving F1

### Current Repository State:

All analysis and fixes are committed on branch:
- `claude/analyze-marl-performance-016azuzUuogdCLSfJf6gMQaJ`

Files available:
- `MARL_Performance_Analysis.md` - Original analysis (11 issues)
- `FIXES_APPLIED.md` - Documentation of fixes applied
- `MARL_STL_BFT_Training_v1_T4_Optimized [FIXED].ipynb` - Training notebook with fixes
- `DATA_SPLIT_ISSUE.md` - This file

---

## Instructions for New Thread

### Prompt to Start New Thread:

```
I need to fix a critical data split issue in my MARL training pipeline.

BACKGROUND:
- Training a MARL system for APT detection on DARPA TC dataset
- Current problem: Validation set has 1% threats but training has 9% threats
- This 10x distribution mismatch prevents the model from learning

TASK:
1. Analyze DARPA_Dataset_Execution_Pipeline.ipynb to find data splitting code
2. Modify to use stratified splitting (all splits should have ~8.9% threats)
3. Regenerate train.parquet, val.parquet, test.parquet files
4. Verify all splits have proper threat distribution
5. Update training config (entropy_coef=0.05, lr=3e-4, value_coef=1.5)
6. Run training and monitor for positive rewards

CURRENT STATE:
- Train: 350k samples, 8.9% threats ✓
- Val: 75k samples, 1.04% threats ✗ (should be 8.9%)
- Test: 75k samples, unknown (likely ~1%)

EXPECTED AFTER FIX:
- All splits: ~8.9% threats (stratified properly)
- Training converges with positive rewards by iteration 100
- F1-score reaches 65-80% by iteration 150

Please help me:
1. Find the splitting code in DARPA_Dataset_Execution_Pipeline.ipynb
2. Add proper stratification using sklearn.train_test_split
3. Verify the fix worked
4. Update training config and run
```

---

## Why This Will Work

The MARL model architecture and reward structure are **fundamentally sound**. The only issue is the data distribution mismatch.

Once validation has the same 8.9% threat rate as training:
- Model can learn a consistent strategy
- Episode rewards will turn positive
- Evaluation metrics will show balanced precision/recall
- The model will actually detect threats (TPs > 0)

**The fix is straightforward - just need to add `stratify` parameter to train_test_split.**

---

## Contact/Context for New Thread

If you need to reference previous analysis:
- See `MARL_Performance_Analysis.md` for full root cause analysis
- See `FIXES_APPLIED.md` for what was already fixed
- Current issue is **NEW** - discovered during training runs

The previous fixes (reward rebalancing, entropy increase, etc.) are still valid and applied. This data split issue is an additional problem that must be fixed separately.

---

**Next Action:** Open new thread with the prompt above and tackle the data pipeline fix.
