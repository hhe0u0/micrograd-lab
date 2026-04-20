# Lab Restructuring Plan

## New Lab Structure

### Lab 1: Numeric Computation in Python (Understanding NumPy Under the Hood)
**Focus:** Pure Python implementation to understand how NumPy works internally

**Content:**
1. Introduction to magic methods and operator overloading
2. Implementing a simple Array class from scratch
   - Element-wise operations (+, -, *, /)
   - Shape validation
   - Broadcasting basics
3. Understanding matrix multiplication
   - Inner workings explained step-by-step
   - Manual implementation
4. Comparison with NumPy
5. **Exercise:** Implement Array class with element-wise ops

**Key Point:** NO NumPy allowed in exercises - students implement from scratch

---

### Lab 2: Computation Graphs and Forward Propagation
**Focus:** Build Value class that tracks operations, introduce NumPy

**Content:**
1. What is a computation graph?
2. The Value class - tracking data and operations
3. Forward pass implementation
4. **NEW: NumPy Basics Section**
   - Basic NumPy operations (array creation, indexing, slicing)
   - Element-wise operations
   - Matrix operations (`np.matmul`, `np.dot`)
   - Broadcasting rules
5. **Exercise 1:** Implement Value class with forward pass
6. **Exercise 2:** Use NumPy for matrix multiplication (given API spec + docs link)
7. Visualize computation graphs

**Key Point:** Students can NOW use NumPy after understanding Lab 1

---

### Lab 3: Gradients and Backward Propagation
**Focus:** Manual backprop using chain rule

**Content:**
1. Calculus refresher - derivatives and chain rule
2. What are gradients and why they matter
3. Manual gradient calculation examples
4. The backward pass
5. **Exercise:** Manually compute gradients for compound expressions
6. Visualize gradients in computation graphs

**Key Point:** Builds directly on Lab 2's Value class

---

### Lab 4: Building and Training a Neural Network
**Focus:** Combine everything - complete micrograd from Karpathy

**Content:**
1. Review: What we've built so far
2. **Provided:** Complete Value class with autograd (from micrograd)
   - Students don't implement topological sort
   - Focus on understanding, not implementing autograd
3. **Exercise:** Implement neural network components
   - Neuron class
   - Layer class
   - MLP (Multi-Layer Perceptron) class
   - Following micrograd/nn.py structure
4. **Training Section:**
   - Loss function (MSE)
   - Training loop
   - Learning rate and optimization
5. **Demo:** Train on 2D dataset (concentric circles from Karpathy's demo.ipynb)
6. **Visualization:** Show decision boundaries
   - Provide expected end-state picture from demo.ipynb
   - Students validate their implementation matches

**Key Point:** Focus on neural network architecture, not autograd internals

---

### Lab 5: Deep Learning with PyTorch (Advanced)
**Focus:** Real-world deep learning with production framework

**Content:**
1. From micrograd to PyTorch - what's the difference?
2. PyTorch basics
   - Tensors vs Value objects
   - Automatic differentiation
   - nn.Module, nn.Linear
3. **Exercise:** Build 2-layer MLP for MNIST
   - Load MNIST from sklearn
   - Define MLP architecture using PyTorch
   - Implement training loop
   - Implement validation loop
   - Track metrics (loss, accuracy)
4. **API References Provided:**
   - torch.nn.Module
   - torch.nn.Linear
   - torch.optim.SGD or Adam
   - torch.nn.functional (loss functions)
5. **Validation:**
   - Plot training curves
   - Evaluate on test set
   - Confusion matrix
6. **Reflection:** How is this different from Lab 4?

**Key Point:** Full PyTorch workflow, students write complete training pipeline

---

## Implementation Checklist

### Lab 1
- [ ] Remove Matrix class, replace with Array class
- [ ] Add detailed explanation of magic methods
- [ ] Add NumPy comparison section
- [ ] Add exercises for element-wise operations
- [ ] Add manual matrix multiplication exercise

### Lab 2
- [ ] Keep Value class exercise
- [ ] Add NumPy basics section (creation, indexing, operations)
- [ ] Add NumPy matrix multiplication exercise
- [ ] Link to NumPy documentation
- [ ] Update visualization to show computation graphs

### Lab 3
- [ ] Keep current structure (mostly fine)
- [ ] Add more explanation before exercises
- [ ] Add calculus refresher section
- [ ] Ensure it builds on Lab 2 Value class

### Lab 4
- [ ] Provide complete Value class (import from engine.py)
- [ ] Remove topological sort exercise
- [ ] Add Neuron/Layer/MLP exercises following nn.py
- [ ] Add training loop from demo.ipynb
- [ ] Add decision boundary visualization
- [ ] Include expected output picture

### Lab 5
- [ ] Create from scratch
- [ ] PyTorch introduction
- [ ] MNIST dataset loading
- [ ] MLP architecture exercise
- [ ] Training loop exercise
- [ ] Validation loop exercise
- [ ] Link to PyTorch documentation
- [ ] Add metric tracking and visualization

---

## Key Principles

1. **Progressive Complexity:** Each lab builds on previous
2. **More Explanation:** Add detailed explanations before exercises
3. **API References:** Link to official docs when introducing new libraries
4. **Visual Feedback:** Use visualization to validate understanding
5. **Hands-On:** Students write code, not just read

---

## References to Include

### Lab 2 (NumPy)
- https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
- https://numpy.org/doc/stable/reference/generated/numpy.dot.html
- https://numpy.org/doc/stable/user/basics.broadcasting.html

### Lab 4 (micrograd)
- https://github.com/karpathy/micrograd/blob/master/demo.ipynb
- https://github.com/karpathy/micrograd/blob/master/micrograd/nn.py

### Lab 5 (PyTorch)
- https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- https://pytorch.org/docs/stable/optim.html
- https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
