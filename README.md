# Simple Neural Networks

从零实现简单神经网络，让人看懂前向传播和反向传播。

不使用 PyTorch / TensorFlow，只用纯 Python 和 NumPy。适合想理解神经网络如何工作的人。

## 仓库内容

- `01_neuron.py` — 单个神经元（感知机）
- `02_mlp.py` — 多层感知机（XOR 问题）
- `03_backprop.md` — 反向传播原理说明

## 快速开始

```bash
pip install numpy
python 01_neuron.py
```

## 学习路径

1. 先跑 `01_neuron.py`，看单个神经元如何做决策
2. 再跑 `02_mlp.py`，看多层网络如何解决 XOR
3. 阅读 `03_backprop.md` 理解梯度如何传播

## 为什么从零写

框架隐藏了太多细节。用 50 行代码写一个能工作的网络，比看 1000 行文档更容易理解。

---

作者：超级大象 | UNSW Mathematics | Princeton PhD Candidate in Neural Networks
