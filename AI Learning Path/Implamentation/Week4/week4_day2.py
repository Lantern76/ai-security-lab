import numpy as np


# Step 1:
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid

    def sigmoid_derivative(self, sig_output):
        return sig_output * (1 - sig_output)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.hidden = self.sigmoid(self.z1)
        self.z2 = np.dot(self.hidden, self.W2) + self.b2
        self.output = self.sigmoid(self.z2)
        return self.output

    def backwards(self, X, y):
        m = len(y)

        output_error = self.output - y
        output_delta = output_error * self.sigmoid_derivative(self.output)

        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden)

        self.gradient_W2 = np.dot(self.hidden.T, output_delta) / m
        self.gradient_W1 = np.dot(X.T, hidden_delta) / m

        self.gradient_b2 = np.mean(output_delta, axis=0, keepdims=True)
        self.gradient_b1 = np.mean(hidden_delta, axis=0, keepdims=True)

        loss = np.mean(output_error**2)
        return loss

    def update(self):
        self.W1 = self.W1 - self.learning_rate * self.gradient_W1
        self.W2 = self.W2 - self.learning_rate * self.gradient_W2
        self.b1 = self.b1 - self.learning_rate * self.gradient_b1
        self.b2 = self.b2 - self.learning_rate * self.gradient_b2

    def fit(self, X, y, epochs=1000, verbose=True):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backwards(X, y)
            self.update()

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict_proba(self, X):
        """Return probability scores"""
        return self.forward(X)

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)


# Data
np.random.seed(42)
normal_bytes = np.random.randn(50) * 1000 + 3000
normal_duration = np.random.randn(50) * 20 + 40
threat_bytes = np.random.randn(50) * 1000 + 7000
threat_duration = np.random.randn(50) * 30 + 100

X = np.column_stack(
    [
        np.concatenate([normal_bytes, threat_bytes]),
        np.concatenate([normal_duration, threat_duration]),
    ]
)
y = np.array([[0]] * 50 + [[1]] * 50)

# shuffle and split
indices = np.arange(len(y))
np.random.shuffle(indices)
split = int(0.8 * len(y))

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Scaling
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_scaled = (X_train - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std


# Create and Train data
model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
model.fit(X_train_scaled, y_train, epochs=1000)

# Evaluate on both sets
train_preds = model.predict(X_train_scaled)
test_preds = model.predict(X_test_scaled)

train_accuracy = np.mean(train_preds == y_train)
test_accuracy = np.mean(test_preds == y_test)

print(f"\nTrain accuracy: {train_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Gap: {train_accuracy - test_accuracy:.2%}")
