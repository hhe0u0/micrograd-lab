# Lab Restructuring Progress

## ✅ Completed

### Lab 1: Numeric Computation in Python ✅
**File:** `lab1_numeric_computation.ipynb`
**Status:** Complete and reviewed

**Content:**
- Part 1: Magic methods explanation with demos
- Part 2: Building SimpleArray class (demo)
- Part 3: Matrix multiplication deep dive
- Exercise: Implement Array class (NO NumPy)
- 5 comprehensive test cases
- Bonus: Compare with NumPy

**Key Achievement:** Students understand NumPy internals through pure Python implementation

---

### Lab 2: Computation Graphs and Forward Propagation ✅
**File:** `lab2_computation_graph.ipynb`
**Status:** Just completed

**Content:**
- Part 1: What are computation graphs?
- Part 2: Building the Value class
- Exercise 1: Implement Value class with operators
- Part 3: Visualizing graphs with `visualize_graph` helper
- Part 4: NumPy introduction (NOW allowed!)
  - Creating arrays
  - Indexing/slicing
  - Element-wise operations
  - Matrix multiplication (@, np.matmul, np.dot)
- Exercise 2: Matrix multiplication with NumPy
- Links to NumPy documentation

**Key Achievement:** 
- Value class tracks computation history
- Visualization using trace helper from micrograd
- NumPy introduced after understanding fundamentals

---

## 🚧 Remaining

### Lab 3: Neural Networks - Forward and Backward Propagation
**Status:** Need to create

**Planned Content:**
- Part 1: Neural network basics
- Part 2: Forward propagation
  - Exercise 1: Implement forward pass for Neuron, Layer
- Part 3: Backward propagation (autograd)
  - Example: Autograd for addition operator
  - Exercise 2: Implement backward pass for each operator (nested autograd)
- Visualization of gradients

**Key Point:** Students implement autograd utilities for each operator

---

### Lab 4: Building and Training a Neural Network
**Status:** Need to create

**Planned Content:**
- Provide complete Value class with autograd (engine.py)
- Exercise: Implement Neuron, Layer, MLP (following micrograd/nn.py)
- Training section with loss function
- Demo: Train on 2D dataset (from demo.ipynb)
- Visualization: Decision boundaries
- Include expected output picture

**Key Point:** Students DON'T implement topological sort - focus on NN architecture

---

### Lab 5: Deep Learning with PyTorch
**Status:** Need to create

**Planned Content:**
- PyTorch introduction
- MNIST dataset from sklearn
- Exercise: Build 2-layer MLP
- Exercise: Implement training loop
- Exercise: Implement validation loop
- Metrics tracking and visualization
- API references for all PyTorch functions

**Key Point:** Full production workflow with PyTorch

---

## Next Steps

1. Create Lab 3 (Neural Networks with manual backprop)
2. Create Lab 4 (Complete NN training with micrograd)
3. Create Lab 5 (PyTorch + MNIST)
4. Update README.md with new lab structure
5. Test all labs in Colab

## Estimated Time Remaining

- Lab 3: ~20 minutes
- Lab 4: ~25 minutes (includes demo.ipynb adaptation)
- Lab 5: ~20 minutes
- Documentation updates: ~10 minutes
- Testing: ~15 minutes

**Total:** ~90 minutes

---

## Files Status

✅ `lab1_numeric_computation.ipynb` - Complete
✅ `lab2_computation_graph.ipynb` - Complete
⏳ `lab3_neural_networks.ipynb` - To create
⏳ `lab4_training_nn.ipynb` - To create
⏳ `lab5_pytorch_mnist.ipynb` - To create
