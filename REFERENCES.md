# References and Attribution

## Primary Inspiration

### Andrej Karpathy's micrograd

This lab series is directly inspired by and builds upon concepts from:

**Repository:** [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)

**Author:** Andrej Karpathy ([@karpathy](https://github.com/karpathy))

**Description:** A tiny scalar-valued autograd engine and a neural network library on top of it with PyTorch-like API

**License:** MIT License

```
MIT License

Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Key Educational Resource

**YouTube Lecture:** [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)

This 2.5-hour lecture walks through building micrograd from scratch, explaining:
- Derivative calculus and chain rule
- Neural network architecture
- Backpropagation algorithm
- Training loops and optimization

**Recommended:** Watch this lecture alongside completing the labs for deeper understanding.

---

## What This Lab Adds

While inspired by micrograd, this lab series provides:

### 1. **Structured Learning Path**
- 5 progressive labs from basics to MLP training
- Each lab builds on previous concepts
- Clear learning objectives for each session

### 2. **Hands-On Exercises**
- Students implement code themselves (not just reading)
- Starter code with TODO comments
- Immediate feedback through validation tests

### 3. **Educational Scaffolding**
- Lab 1: Python magic methods (foundation)
- Lab 2: Value object and computation graphs
- Lab 3: Manual backpropagation (understanding chain rule)
- Lab 4: Automatic backpropagation (implementing autograd)
- Lab 5: Training neural networks (complete system)

### 4. **Additional Learning Aids**
- Worked examples with detailed explanations
- Edge case testing for deeper understanding
- Collapsible answer sections for self-guided learning
- Visual computation graph examples
- Matrix operations warm-up (Lab 1)

### 5. **Test-Driven Learning**
- Each exercise includes validation tests
- Immediate feedback on correctness
- Edge cases to catch common mistakes
- Encourages iterative development

---

## Core Concepts from Micrograd

The following concepts are directly from Karpathy's micrograd:

### 1. **Value Class**
```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
```

**Key Ideas:**
- Scalar values that track their computation history
- Gradient accumulation for backpropagation
- Closure-based backward functions

### 2. **Operator Overloading**
Magic methods for arithmetic operations:
- `__add__`, `__mul__`, `__pow__` for basic operations
- `__radd__`, `__rmul__` for reverse operations
- `__neg__`, `__sub__`, `__truediv__` for convenience

### 3. **Automatic Differentiation**
Topological sort through computation graph:
```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

### 4. **Neural Network Architecture**
Building blocks for neural networks:
- `Neuron`: Single neuron with weights and bias
- `Layer`: Collection of neurons
- `MLP`: Multi-layer perceptron (stack of layers)
- `Module`: Base class with `parameters()` method

### 5. **Training Loop**
Standard ML training pattern:
```python
# Forward pass
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

# Backward pass
for p in mlp.parameters():
    p.grad = 0.0
loss.backward()

# Update parameters
for p in mlp.parameters():
    p.data += -0.01 * p.grad
```

---

## Additional Educational Resources

### Deep Learning Fundamentals

1. **CS231n: Convolutional Neural Networks for Visual Recognition**
   - Stanford course by Andrej Karpathy and others
   - [cs231n.stanford.edu](http://cs231n.stanford.edu/)

2. **Deep Learning Book**
   - Goodfellow, Bengio, and Courville
   - [deeplearningbook.org](https://www.deeplearningbook.org/)

3. **Neural Networks: Zero to Hero**
   - Karpathy's YouTube series
   - Includes micrograd, makemore, and GPT implementations

### Automatic Differentiation

1. **Automatic Differentiation in Machine Learning: a Survey**
   - Baydin, Pearlmutter, Radul, Siskind (2018)
   - Comprehensive overview of autodiff techniques

2. **PyTorch Autograd**
   - Official PyTorch documentation
   - [pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

### Backpropagation

1. **Calculus on Computational Graphs: Backpropagation**
   - Christopher Olah's blog post
   - [colah.github.io/posts/2015-08-Backprop/](https://colah.github.io/posts/2015-08-Backprop/)

2. **Yes you should understand backprop**
   - Andrej Karpathy's blog post
   - [karpathy.github.io/2016/05/31/rl/](https://karpathy.github.io/2016/05/31/rl/)

---

## How to Use This Lab with Original Micrograd

### Recommended Learning Path

1. **Start with Lab 1-3** of this workshop
   - Build foundation in magic methods
   - Understand forward pass and chain rule
   - Practice manual backpropagation

2. **Watch Karpathy's YouTube lecture**
   - See the complete implementation walkthrough
   - Understand design decisions
   - Get intuition for autodiff

3. **Complete Lab 4-5** of this workshop
   - Implement automatic backpropagation
   - Build complete neural network
   - Train on real dataset

4. **Explore original micrograd repository**
   - Study the production implementation
   - Read the code comments
   - Try the demo notebooks

5. **Extend your implementation**
   - Add more operations (tanh, exp, log)
   - Implement optimizers (Adam, RMSprop)
   - Visualize computation graphs
   - Add batch operations

### Comparing Implementations

After completing the labs, compare your implementation with micrograd:

**File Correspondence:**
- `engine.py` (this lab) ↔ `micrograd/engine.py` (original)
- `nn.py` (this lab) ↔ `micrograd/nn.py` (original)
- Lab 5 training loop ↔ `demo.ipynb` (original)

**Key Differences:**
- This lab uses more explicit variable names for learning
- Original micrograd is more concise and production-ready
- This lab includes extensive comments and validation tests
- Original has more operations (tanh, exp, sigmoid)

---

## Citation

If you use this lab material or build upon it, please cite both this lab and the original micrograd:

```bibtex
@misc{micrograd,
  author = {Karpathy, Andrej},
  title = {micrograd: A tiny scalar-valued autograd engine},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/karpathy/micrograd}},
}

@misc{micrograd_lab,
  title = {Micrograd Lab: Interactive Neural Network Course},
  year = {2024},
  note = {Educational workshop based on Karpathy's micrograd},
  howpublished = {Interactive Jupyter notebooks}
}
```

---

## License

This lab material is provided under the MIT License, consistent with the original micrograd.

The core concepts, algorithms, and design patterns are from Andrej Karpathy's micrograd (MIT License, Copyright 2020).

The educational scaffolding, exercises, and additional content are original contributions to help learners understand these concepts.

---

## Questions or Issues?

- **For micrograd conceptual questions:** Refer to [Karpathy's lecture](https://www.youtube.com/watch?v=VMj-3S1tku0)
- **For original micrograd code:** See [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- **For this lab's exercises:** Check the notebook cells and validation tests

---

**Last Updated:** April 2024

**Acknowledgment:** Special thanks to Andrej Karpathy for making deep learning accessible through clear explanations and open-source implementations.
