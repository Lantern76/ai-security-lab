# Week 3 Day 8: Review + Syntax Test

## Day 8 Mission

1. Tie all Week 3 concepts together
2. Syntax accuracy test
3. Identify gaps
4. Prepare for Week 4

---

## Week 3 Concept Review

### Day 1: What is Learning?
- Learning = Finding parameters that minimize error
- Attack surface: Data, loss, gradients, parameters

### Day 2: Loss Functions
- MSE: Sensitive to outliers
- MAE: Robust to outliers
- BCE: For classification, penalizes confident mistakes

### Day 3: Gradient Descent
- Gradient = Direction of steepest increase
- Update: w = w - lr * gradient
- FGSM: Use gradient to create adversarial examples

### Day 4: Training Loop
- Forward → Loss → Gradient → Update
- Every component is an attack surface
- Backdoor injection during training

### Day 5: Linear Regression
- y = X @ w + b
- Model extraction: n queries = exact weights
- Defense: Rate limiting, output noise

### Day 6: Logistic Regression
- Sigmoid squashes to probability
- BCE loss for classification
- Threshold gaming attacks

### Day 7: Model Evaluation
- Confusion matrix: TP, TN, FP, FN
- Precision: Of predicted positives, how many correct?
- Recall: Of actual positives, how many caught?
- Evaluation manipulation attacks

---

## Flow: Complete ML Pipeline

```
Raw Data
    ↓
Preprocess (clean, scale, encode) ← Poisoning attacks
    ↓
Split (train/test)
    ↓
Initialize (random weights)
    ↓
Training Loop:                    ← Gradient attacks, backdoors
    Forward pass
    Calculate loss               ← Loss manipulation
    Calculate gradient
    Update weights
    ↓
Trained Model                    ← Model extraction
    ↓
Evaluate (precision, recall, F1) ← Evaluation manipulation
    ↓
Deploy                           ← Adversarial examples
```

---

## Syntax Test: Cold Recall

### Section 1: Core Python
Type without looking:

1. For loop with enumerate
2. Function that returns a value
3. Accumulator pattern (empty list, append, return)

### Section 2: NumPy
Type without looking:

1. Create array
2. Dot product
3. Mean and standard deviation
4. MSE calculation

### Section 3: Pandas
Type without looking:

1. Filter rows where column equals value
2. Group by and size
3. Fill NA with median
4. One-hot encode

### Section 4: Training Loop
Type without looking:

1. Initialize weights randomly
2. Forward pass (prediction)
3. MSE loss
4. Gradient calculation
5. Weight update

### Section 5: Classification
Type without looking:

1. Sigmoid function
2. BCE loss (simplified)
3. Binary prediction from probability
4. Precision formula
5. Recall formula

---

## Syntax Accuracy Scoring

After completing cold recall:

| Section | Patterns | Correct | Accuracy |
|---------|----------|---------|----------|
| Core Python | 3 | /3 | % |
| NumPy | 4 | /4 | % |
| Pandas | 4 | /4 | % |
| Training | 5 | /5 | % |
| Classification | 5 | /5 | % |
| **Total** | **21** | **/21** | **%** |

**Target: 80%+ accuracy (17/21 correct)**

---

## Gap Identification

### Concepts: Rate 1-5

| Concept | Confidence | Needs Review? |
|---------|------------|---------------|
| Learning as optimization | /5 | |
| MSE vs MAE vs BCE | /5 | |
| Gradient descent | /5 | |
| Training loop components | /5 | |
| Linear regression | /5 | |
| Logistic regression | /5 | |
| Sigmoid function | /5 | |
| Precision/Recall/F1 | /5 | |

### Security: Rate 1-5

| Attack Type | Confidence | Needs Review? |
|-------------|------------|---------------|
| Data poisoning | /5 | |
| Gradient attacks (FGSM) | /5 | |
| Model extraction | /5 | |
| Backdoor injection | /5 | |
| Evaluation manipulation | /5 | |

---

## Project: End-to-End Pipeline

Build a complete threat classifier:

```python
import numpy as np
import pandas as pd

# 1. Generate synthetic threat data
# 2. Preprocess (scale, encode)
# 3. Split train/test
# 4. Train logistic regression from scratch
# 5. Evaluate (confusion matrix, precision, recall, F1)
# 6. Test at multiple thresholds
# 7. Identify which threshold is best for security use case
```

This combines everything from Week 3.

---

## Week 3 Deliverables Checklist

- [ ] Gradient descent implementation
- [ ] Linear regression from scratch (class)
- [ ] Logistic regression from scratch (class)
- [ ] Evaluation function (all metrics)
- [ ] Complete pipeline project
- [ ] Syntax test: 80%+ accuracy

---

## Week 4 Preview

**Topic:** Neural Networks

- Why deep learning?
- Backpropagation
- Activation functions
- Multi-layer networks
- Adversarial robustness

**Syntax continues:** 
- More complex patterns
- Class structures
- Matrix operations at scale

---

## Week 3 Summary

**Concepts Mastered:**
- Learning as optimization
- Loss functions and their vulnerabilities
- Gradient descent mechanics
- Training loop anatomy
- Linear and logistic regression
- Model evaluation metrics

**Security Awareness:**
- Every ML component is an attack surface
- Training time vs inference time attacks
- Evaluation can be manipulated
- Defense requires multiple layers

**Syntax Progress:**
- Week 2 end: ~65%
- Week 3 target: 80-85%
- Actual: [To be measured]

---

## Final Assessment

After completing all reviews and tests:

| Dimension | Score | Target | Met? |
|-----------|-------|--------|------|
| Concepts | /5 | 4+ | |
| Security | /5 | 4+ | |
| Syntax | % | 80%+ | |
| Projects | /3 | 3 | |
