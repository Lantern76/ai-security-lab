# Week 3: Machine Learning Fundamentals + Syntax Push

## Week 3 Mission

**Dual focus:**
1. Learn ML fundamentals (what is learning, loss functions, gradient descent)
2. Push syntax from 65 → 80-85

**Philosophy:** Fewer concepts, deeper mastery. Every pattern drilled until automatic.

---

## Updated Daily Structure

| Block | Time | Purpose |
|-------|------|---------|
| Check-in | 2 min | Energy, state, readiness |
| Warm-up | 10 min | Cold recall - all previous patterns |
| Brain Integration Exercise | 5 min | AI adaptation training |
| Lesson | 45-60 min | New concepts + parallel security threads |
| Syntax Drilling | 20 min | Every pattern 3-5x |
| Project | 30 min | Build something - you type everything |
| Rapid Fire | 15 min | Speed round - no thinking, just typing |
| Breakdown | 5 min | Assessment + tomorrow preview |

**Total: ~2.5 hours per day**

---

## Week 3 Topics

| Day | Primary Topic | Security Thread | Syntax Focus |
|-----|---------------|-----------------|--------------|
| 1 | What is Learning? | What is adversarial learning? | Core Python patterns |
| 2 | Loss Functions | How attackers exploit loss | NumPy patterns |
| 3 | Gradient Descent | Gradient-based attacks | Pandas patterns |
| 4 | Training Loop from Scratch | Training-time attacks | Z-score pattern |
| 5 | Linear Regression | Model extraction | Full pipeline |
| 6 | Logistic Regression | Classification attacks | Full pipeline |
| 7 | Model Evaluation | Evaluation manipulation | All patterns |
| 8 | Week 3 Review | Integration | Syntax test |

---

## Syntax Targets

### Patterns to Master by End of Week 3

**Core Python (must be instant):**
```python
for item in items:
for i, item in enumerate(items):
if condition:
def function(param):
    return result
results = []
results.append(item)
```

**NumPy (must be instant):**
```python
import numpy as np
np.array([1, 2, 3])
np.dot(a, b)
np.mean(data)
np.std(data)
np.abs(x)
np.where(condition)[0]
data.shape
```

**Pandas (must be instant):**
```python
import pandas as pd
df = pd.DataFrame(data)
df["column"]
df[df["col"] == value]
df[(cond1) & (cond2)]
df.groupby("col").size()
df["col"].fillna(df["col"].median())
df.dropna(subset=["col"])
pd.get_dummies(df, columns=["col"])
df.sort_values("col", ascending=False)
```

**Z-Score (must be instant):**
```python
mean = df["col"].mean()
std = df["col"].std()
df["col"] = (df["col"] - mean) / std
```

---

## Syntax Drilling Rules

1. **Every new pattern:** Type it 5x before moving on
2. **Every error:** Retype the whole line correctly 3x
3. **No copy-paste:** You type everything
4. **End of day:** All patterns from that day, rapid fire
5. **Start of next day:** All patterns from previous day, cold

---

## Success Metrics

### Syntax (Primary Focus)
- [ ] Core Python patterns: No hesitation
- [ ] NumPy patterns: No errors
- [ ] Pandas patterns: Automatic
- [ ] Z-score pattern: Instant recall
- [ ] Accuracy: 80-85%

### Concepts (Secondary Focus)
- [ ] Can explain "what is learning" mathematically
- [ ] Understand loss functions conceptually
- [ ] Can implement gradient descent from scratch
- [ ] See security implications at each step

---

## Projects

| Day | Project |
|-----|---------|
| 4 | Training loop from scratch |
| 6 | Binary classifier for threat detection |
| 7 | Complete ML pipeline with evaluation |

---

## Week 3 Deliverables

By end of Week 3:
1. Working gradient descent implementation
2. Working linear regression from scratch
3. Working logistic regression from scratch
4. Syntax accuracy at 80-85%
5. All patterns typed without hesitation
