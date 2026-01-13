# === WEEK 3 DAY 7: COMPLETE SUPERVISED LEARNING PIPELINE ===
# Train/Test Split Methodology for Proper Model Evaluation
# Imports
import numpy as np

# Step 1: Create Data
np.random.seed(42)

# Create normal samples
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40

# Create threat samples
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 30 + 100

# Combine into X and y
bytes = np.concatenate([normal_bytes, threat_bytes])
duration = np.concatenate([normal_duration, threat_duration])
X = np.column_stack([bytes, duration])
y = np.array([0] * 50 + [1] * 50)


# Step 2: Shuffle and Split
n_samples = len(y)
indices = np.arange(n_samples)

np.random.shuffle(indices)

X_shuffled = X[indices]
y_shuffled = y[indices]

split = int(0.8 * n_samples)

X_train = X_shuffled[:split]  # Rows 0-79 (first 80)
y_train = y_shuffled[:split]

X_test = X_shuffled[split:]  # Rows 80-99 (last 20)
y_test = y_shuffled[split:]

# Step 3: Scale the data (using ONLY training statistics)
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std


# Step 4: LogisticRegression
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


# Step 5: Evaluation Functions
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN


def precision_recall(TP, TN, FP, FN):
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def f1_score(precision, recall):
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# Step 6: Train and Evaluate
model = LogisticRegression(learning_rate=0.01)
model.fit(X_train_scaled, y_train, epochs=1000)

train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

# Evaluate on training data
train_TP, train_TN, train_FP, train_FN = confusion_matrix(y_train, train_preds)
train_precision, train_recall = precision_recall(train_TP, train_TN, train_FP, train_FN)
train_f1 = f1_score(train_precision, train_recall)

# Evaluate on test data
test_TP, test_TN, test_FP, test_FN = confusion_matrix(y_test, test_preds)
test_precision, test_recall = precision_recall(test_TP, test_TN, test_FP, test_FN)
test_f1 = f1_score(test_precision, test_recall)

print("\n" + "=" * 50)
print("=== Training Performance ===")
print(f"Precision: {train_precision:.2f}")
print(f"Recall: {train_recall:.2f}")
print(f"F1 Score: {train_f1:.2f}")

print("\n=== Test Performance ===")
print(f"Precision: {test_precision:.2f}")
print(f"Recall: {test_recall:.2f}")
print(f"F1 Score: {test_f1:.2f}")

print("\n=== Overfitting Analysis ===")
print(f"F1 Gap (Train - Test): {train_f1 - test_f1:.2f}")
print(f"Recall Gap (Train - Test): {train_recall - test_recall:.2f}")
