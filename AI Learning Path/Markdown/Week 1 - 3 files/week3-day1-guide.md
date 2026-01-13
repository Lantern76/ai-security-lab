# Week 3 Day 1: What is Learning?

## Learning Objectives
- Understand learning as optimization
- Mathematical definition of "learning"
- Connection to adversarial learning
- Core Python syntax drilled to automaticity

---

## Concept: What is Learning?

### The Core Insight

Learning = Finding parameters that minimize error

```
Data → Model(parameters) → Predictions → Compare to truth → Adjust parameters → Repeat
```

### Mathematical Definition

Given:
- Data: X (inputs), y (true outputs)
- Model: f(X, w) where w = parameters/weights
- Error: How wrong predictions are

**Learning:** Find w that minimizes error

```
w* = argmin Loss(f(X, w), y)
```

Translation: Find the weights (w*) that make the loss as small as possible.

---

## Security Thread: What is Adversarial Learning?

If learning = minimizing error on data...

**Adversarial learning = manipulating that process**

Attack vectors:
1. **Poison the data** → Model learns wrong patterns
2. **Manipulate the loss** → Model optimizes for wrong objective
3. **Attack the parameters** → Directly corrupt learned weights
4. **Fool the predictions** → Input designed to cause wrong output

**Key insight:** Every component of learning is an attack surface.

---

## Syntax Drilling: Core Python

### Pattern 1: For Loop
```python
for item in items:
    print(item)
```
*Type 5x*

### Pattern 2: Enumerate
```python
for i, item in enumerate(items):
    print(i, item)
```
*Type 5x*

### Pattern 3: Conditional
```python
if score > threshold:
    print("Alert")
else:
    print("Normal")
```
*Type 5x*

### Pattern 4: Function Definition
```python
def calculate_score(data, weights):
    result = np.dot(data, weights)
    return result
```
*Type 5x*

### Pattern 5: Accumulator Pattern
```python
results = []
for item in items:
    if item > threshold:
        results.append(item)
return results
```
*Type 5x*

---

## Exercises

### Exercise 1: Explain Learning
In your own words, what does it mean for a model to "learn"?

### Exercise 2: Attack Surface Mapping
List 3 ways an attacker could manipulate the learning process.

### Exercise 3: Syntax Cold Recall
Without looking, type:
1. A for loop with enumerate
2. A function that returns a value
3. The accumulator pattern (empty list, loop, append)

---

## Project: Simple Learning Demonstration

Build a program that:
1. Has "true" weights: `[2, 3]`
2. Generates predictions: `y = X @ true_weights`
3. Starts with random guess weights
4. Measures error (how far off)
5. Shows that correct weights = zero error

This demonstrates: Learning = finding weights that minimize error.

---

## Rapid Fire Syntax (End of Day)

Type each 3x, no looking:

1. `for item in items:`
2. `for i, item in enumerate(items):`
3. `if condition:`
4. `def func(param):`
5. `return result`
6. `results = []`
7. `results.append(item)`

---

## Checklist

- [ ] Can explain learning as optimization
- [ ] Identified 3 adversarial attack vectors
- [ ] Core Python patterns typed without errors
- [ ] Project completed
- [ ] Rapid fire completed
