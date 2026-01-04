# Todays lesson is introducing accuracy  and predictability to the model
# Imports
import numpy as np


# Step 1: Connfusion Matrix
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


# Step 2: Define precision_recall function
def precision_recall(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


# Step 3: Test data
y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])

# Step 4: Call Functions
TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

precision, recall = precision_recall(TP, TN, FP, FN)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


# Step 5: F1 score
def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


print(f1_score(precision, recall))


# Live project connecting previous data

# Step 1: Logitistic Regression


class LogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.weights = None

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def fit(self, X, y, epochs=1000):
        self.weights = np.random.randn(X.shape[1])

        for i in range(epochs):
            z = np.dot(X, self.weights)
            predictions = self.sigmoid(z)
            error = predictions - y
            loss = np.mean((predictions - y) ** 2)
            gradient = np.dot(X.T, error) / len(y)
            self.weights = self.weights - self.learning_rate * gradient

            if i % 100 == 0:
                print(f"Epoch {i}: loss = {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights)
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)


# Step 2: Training Data
np.random.seed(42)

# Create normal samples
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40

# Create threat samples
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) + 20 + 100

# Combine
bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([0] * 50 + [1] * 50)

# test code
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 3: Scale and Train
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Make model
model = LogisticRegression(learning_rate=0.01)
model.fit(X_scaled, y, epochs=1000)

# Test
print("Learned Weights:", model.weights)


# Step 4: Evaluate with Matrix

# Get predictions
predictions = model.predict(X_scaled)

# Calculate confusion matrix
TP, TN, FP, FN = confusion_matrix(y, predictions)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Calculate precision, recall, F1
precision, recall = precision_recall(TP, TN, FP, FN)
f1 = f1_score(precision, recall)

print(f"Precision: {precision:.2f}")
print(f"Recall:{recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 5: Make scenerio more difficult

# harder data - more overlap
np.random.seed(42)

# normal traffic (label 0)
threat_bytes = np.random.randn(50) * 2000 + 6000
threat_duration = np.random.randn(50) * 30 + 80

# Rebuild X and Y
bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([0] * 50 + [1] * 50)

# Scale
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Retrain
model = LogisticRegression(learning_rate=0.1)
model.fit(X_scaled, y, epochs=1000)

# Evaluate
predictions = model.predict(X_scaled)
TP, TN, FP, FN = confusion_matrix(y, predictions)
precision, recall = precision_recall(TP, TN, FP, FN)
f1 = f1_score(precision, recall)

print(f"\nTP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Step 6: Adjust threshold to catch missing True Possitives

# Get probabiliites
probabilities = model.predict_proba(X_scaled)

# Use lower threshold
predictions_low = (probabilities > 0.3).astype(int)

# Evaluate
TP, TN, FP, FN = confusion_matrix(y, predictions_low)
precision, recall = precision_recall(TP, TN, FP, FN)
f1 = f1_score(precision, recall)

print("Threshold 0.3:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Lower threshold mean more false possitives, but captured more true possitives as a result.
