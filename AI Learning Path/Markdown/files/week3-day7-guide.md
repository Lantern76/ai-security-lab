# Week 3 Day 7: Model Evaluation

## Learning Objectives
- Evaluation metrics beyond accuracy
- Confusion matrix and its components
- Precision, recall, F1-score
- How attackers manipulate evaluation

---

## Why Accuracy Isn't Enough

### The Imbalanced Data Problem

Dataset: 99 normal, 1 threat

Model predicts everything as "normal":
- Accuracy = 99%
- But missed the only threat!

**For security:** Missing threats (false negatives) is often worse than false alarms.

---

## Confusion Matrix

```
                Predicted
              Normal  Threat
Actual Normal   TN      FP
       Threat   FN      TP
```

| Term | Meaning |
|------|---------|
| TP (True Positive) | Correctly identified threat |
| TN (True Negative) | Correctly identified normal |
| FP (False Positive) | Normal flagged as threat (false alarm) |
| FN (False Negative) | Threat missed (dangerous!) |

---

## Key Metrics

### Precision
"Of all predicted threats, how many were real?"
```python
precision = TP / (TP + FP)
```
High precision = Few false alarms

### Recall (Sensitivity)
"Of all real threats, how many did we catch?"
```python
recall = TP / (TP + FN)
```
High recall = Few missed threats

### F1-Score
Harmonic mean of precision and recall:
```python
f1 = 2 * (precision * recall) / (precision + recall)
```

---

## The Precision-Recall Tradeoff

| Threshold | Precision | Recall |
|-----------|-----------|--------|
| Low (flag more) | Lower | Higher |
| High (flag less) | Higher | Lower |

**Security choice:** Usually favor recall (catch threats) over precision (reduce false alarms).

---

## Security Thread: Evaluation Manipulation

### Attack 1: Test Set Poisoning
- Attacker knows evaluation data
- Poisons test set to make bad model look good
- Model passes evaluation but fails in production

### Attack 2: Metric Gaming
- Attacker optimizes for your metric
- If you measure accuracy, they target the majority class
- If you measure recall, they cause false positives

### Attack 3: Distribution Shift
- Model trained on one distribution
- Attacker shifts real-world distribution
- Evaluation no longer reflects reality

### Defense: Multiple Metrics
- Never rely on single metric
- Use confusion matrix
- Test on held-out data attacker can't access
- Monitor production performance vs evaluation

---

## Syntax Drilling: Evaluation Patterns

### Pattern 1: Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
```
*Type 5x*

### Pattern 2: Manual Precision
```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
```
*Type 5x*

### Pattern 3: Manual Recall
```python
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
```
*Type 5x*

### Pattern 4: Manual F1
```python
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
```
*Type 5x*

### Pattern 5: Classification Report
```python
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))
```
*Type 5x*

### Pattern 6: All Metrics Function
```python
def evaluate_classifier(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```
*Type 3x*

---

## Exercises

### Exercise 1: Confusion Matrix
Given: y_true = [1,1,0,0,1,0,1,0], y_pred = [1,0,0,1,1,0,0,0]
Calculate TP, TN, FP, FN by hand.

### Exercise 2: Metric Calculation
Using your confusion matrix from Exercise 1:
- Calculate precision
- Calculate recall
- Calculate F1

### Exercise 3: Threshold Selection
Your threat detector has:
- At threshold 0.3: Precision=0.6, Recall=0.95
- At threshold 0.7: Precision=0.9, Recall=0.5

Which do you choose for a nuclear facility? For a spam filter? Why?

---

## Project: Complete Evaluation Pipeline

Build:
```python
def full_evaluation(model, X_test, y_test, thresholds=[0.3, 0.5, 0.7]):
    """
    Evaluate model at multiple thresholds.
    Return confusion matrix and metrics for each.
    """
    results = {}
    for threshold in thresholds:
        probs = model.predict_proba(X_test)
        preds = (probs > threshold).astype(int)
        
        # Calculate all metrics
        # Store in results[threshold]
    
    return results
```

Apply to your threat classifier from Day 6.
Show how metrics change with threshold.

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `tn, fp, fn, tp = cm.ravel()`
2. `precision = tp / (tp + fp)`
3. `recall = tp / (tp + fn)`
4. `f1 = 2 * (precision * recall) / (precision + recall)`
5. `np.sum((y_true == 1) & (y_pred == 1))`
6. `from sklearn.metrics import confusion_matrix`
7. `print(classification_report(y_true, y_pred))`

---

## Checklist

- [ ] Understand why accuracy isn't enough
- [ ] Can calculate confusion matrix components
- [ ] Know precision vs recall tradeoff
- [ ] Understand evaluation manipulation attacks
- [ ] Evaluation patterns typed without errors
- [ ] Project completed (multi-threshold evaluation)
