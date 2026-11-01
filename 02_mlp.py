"""
多层感知机 — 解决 XOR 问题
XOR 是单层感知机无法解决的经典问题
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

class MLP:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.5
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, x):
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            z = activations[-1] @ w + b
            a = sigmoid(z)
            activations.append(a)
        return activations
    
    def train(self, X, y, epochs=5000, lr=0.5):
        for epoch in range(epochs):
            acts = self.forward(X)
            # 反向传播
            error = acts[-1] - y
            for i in range(len(self.weights) - 1, -1, -1):
                delta = error * sigmoid_deriv(acts[i+1])
                self.weights[i] -= lr * acts[i].T @ delta
                self.biases[i] -= lr * np.mean(delta, axis=0, keepdims=True)
                if i > 0:
                    error = delta @ self.weights[i].T
            if epoch % 1000 == 0:
                loss = np.mean((acts[-1] - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# XOR 数据
X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

mlp = MLP([2, 4, 1])
mlp.train(X, y)

print("\nXOR 预测：")
for x in X:
    out = mlp.forward(x.reshape(1, -1))[-1][0, 0]
    print(f"  {x} → {out:.3f} → {int(out > 0.5)}")
