# GradNet

A neural network library built from scratch in Python, implementing automatic differentiation, computational graphs, and a multilayer perceptron (MLP).

This project explores the fundamental concepts behind neural networks by implementing the core components manually, without relying on deep learning frameworks.

## Features

- **Automatic Differentiation:** Computes gradients automatically using reverse-mode differentiation.
- **Computational Graphs:** Builds computational graphs to track mathematical operations and their dependencies.
- **Backpropagation:** Implements gradient propagation through the computational graph using the chain rule.
- **Mathematical Operations:** Supports basic arithmetic operations, exponentiation, and activation functions such as `tanh`.
- **Neural Network Components:** Implements neurons, layers, and multilayer perceptrons (MLPs).
- **Gradient Descent:** Trains neural networks by updating weights and biases using calculated gradients.

## Project Structure

```text
GradNet/
│
├── GradNet/
│   ├── __init__.py
│   ├── engine.py
│   └── nn.py
│
├── notebooks/
│   └── demo.ipynb
│
├── .gitignore
└── README.md
```

### Core Modules

| File | Description |
|---|---|
| `engine.py` | Implements the `Value` class, mathematical operations, computational graphs, and backpropagation. |
| `nn.py` | Implements the `Neuron`, `Layer`, and `MLP` classes, along with parameter management. |
| `demo.ipynb` | Demonstrates the implementation, forward pass, gradient calculation, and neural network training. |

## Getting Started

### Requirements

- Python
- Jupyter Notebook (for running the demonstration notebook)

### Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
cd GradNet
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### Import the Modules

You can import the implemented classes into your Python code:

```python
from GradNet.engine import Value
from GradNet.nn import Neuron, Layer, MLP
```

### Create a Neural Network

For example, create a multilayer perceptron with:

- 3 input features
- 4 neurons in the first hidden layer
- 4 neurons in the second hidden layer
- 1 output neuron

```python
model = MLP(3, [4, 4, 1])
```

Run a forward pass:

```python
x = [2.0, 3.0, -1.0]

output = model(x)

print(output.data)
```

## Learning Objectives

The main objectives of this project are to understand:

- How computational graphs represent mathematical expressions.
- How derivatives are calculated using the chain rule.
- How gradients flow backward through a neural network.
- How individual neurons combine weights, inputs, and biases.
- How multiple layers form a neural network.
- How gradient descent updates model parameters during training.

## License

This project is intended for educational and learning purposes.