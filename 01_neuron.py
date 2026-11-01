"""
单个神经元（感知机）
用于二分类：根据输入决定输出 0 或 1
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()
    
    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        return sigmoid(z)

# 示例：AND 逻辑
# 输入 [0,0] [0,1] [1,0] [1,1]
# 输出  0    0    0    1

neuron = Neuron(2)
# 手动设置权重，让它能正确识别 AND
neuron.weights = np.array([10.0, 10.0])
neuron.bias = -15.0

test_inputs = [
    np.array([0, 0]),
    np.array([0, 1]),
    np.array([1, 0]),
    np.array([1, 1]),
]

print("AND 门结果：")
for x in test_inputs:
    out = neuron.forward(x)
    print(f"  {x} → {out:.3f} → {int(out > 0.5)}")
