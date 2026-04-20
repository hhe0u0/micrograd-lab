# Micrograd Lab - Interactive Neural Network Course

A hands-on Jupyter notebook series that teaches you how to build a neural network from scratch, step by step.

## 🎯 What You'll Learn

Build your own automatic differentiation engine and neural network library, understanding the core principles behind modern deep learning frameworks like PyTorch and TensorFlow.

## 📚 Lab Structure

### Lab 1: Numeric Computation in Python
**Focus:** Understanding how NumPy works by building arrays from scratch  
**Concepts:** Magic methods, operator overloading, matrix multiplication  
**Exercise:** Implement Array class with element-wise and matrix operations (NO NumPy allowed)  
**What You'll Learn:** How libraries like NumPy implement numeric operations internally

### Lab 2: Computation Graphs and Forward Propagation
**Focus:** Build Value class, understand gradients, learn NumPy  
**Concepts:** Computation graphs, automatic differentiation, gradient calculation  
**Exercises:**
- Exercise 1: Implement Value class that tracks operations
- Exercise 2: Matrix multiplication with NumPy
- Exercise 3: 2D linear regression with gradient calculation  
**What You'll Learn:** Foundation of autograd, now allowed to use NumPy!

### Lab 3: Building and Training Neural Networks
**Focus:** Complete neural network with training (combines original Lab 3 & 4)  
**Concepts:** Neuron, Layer, MLP architecture, loss functions, training loops  
**Given:** Complete Value class with autograd from micrograd  
**Exercises:**
- Exercise 1: Implement Neuron, Layer, MLP classes (especially `__call__` methods)
- Exercise 2: Implement loss function with L2 regularization
- Exercise 3: Implement training loop  
**Dataset:** 2D moon classification (from Karpathy's demo.ipynb)  
**What You'll Learn:** How to build and train neural networks from scratch

### Lab 4: Deep Learning with PyTorch
**Focus:** Production deep learning on real data  
**Concepts:** PyTorch tensors, nn.Module, optimizers, validation  
**Dataset:** MNIST handwritten digits (sklearn version)  
**Exercises:**
- Exercise 1: Build 2-layer MLP using PyTorch
- Exercise 2: Implement training loop with Adam optimizer
- Exercise 3: Implement validation loop  
**What You'll Learn:** How to use PyTorch for real-world deep learning tasks

### Lab 5: Transfer Learning - Cats vs Dogs
**Focus:** Modern deep learning with pre-trained models  
**Concepts:** Transfer learning, feature extraction, data augmentation, evaluation metrics  
**Dataset:** Kaggle Dogs vs Cats (25,000 images)  
**Exercises:**
- Exercise 1: Build model with pre-trained ResNet18 + custom head
- Exercise 2: Implement training loop with BCEWithLogitsLoss
- Exercise 3: Implement validation with precision/recall/F1
- Exercise 4: Analyze predictions and confidence scores  
**What You'll Learn:** How to leverage pre-trained models for new tasks with limited data

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Local Setup

```bash
# Install Jupyter if not already installed
pip install notebook

# Clone or download this repository
cd micrograd-lab

# Install graphviz for computation graph visualization
pip install graphviz

# Start Jupyter
jupyter notebook
```

### Google Colab Setup

**No installation needed!** Google Colab has all required packages pre-installed:
- ✅ graphviz (for computation graph visualization)
- ✅ matplotlib (for plotting)
- ✅ numpy, pandas (for data manipulation)

Simply open any notebook in Colab and start learning!

### Running Labs

Open each lab notebook in order:
1. `lab1_numeric_computation.ipynb` - Understand NumPy internals
2. `lab2_computation_graph.ipynb` - Computation graphs + gradients + NumPy
3. `lab3_neural_networks.ipynb` - Build and train MLP from scratch
4. `lab4_pytorch_mnist.ipynb` - PyTorch on real data (MNIST)
5. `lab5_transfer_learning.ipynb` - Transfer learning (Cats vs Dogs)

Each notebook contains:
- ✅ Clear introduction and concepts
- ✅ Worked demos with visual examples
- ✅ Hands-on exercises with starter code
- ✅ Correct answers (hidden in collapsible sections)
- ✅ Validation tests to check your work
- ✅ Edge cases for deeper understanding

## 📁 Project Structure

```
micrograd-lab/
├── README.md                        # This file
├── REFERENCES.md                    # Attribution to Karpathy's micrograd
├── lab1_numeric_computation.ipynb  # Lab 1: Pure Python arrays
├── lab2_computation_graph.ipynb    # Lab 2: Value class + gradients
├── lab3_neural_networks.ipynb      # Lab 3: Build and train MLP
├── lab4_pytorch_mnist.ipynb        # Lab 4: PyTorch + MNIST
├── engine.py                        # Complete Value class (reference)
├── nn.py                            # Neural network layers (reference)
└── utils.py                         # Helper functions (topological sort, visualization)
```

## 🎓 Learning Philosophy

- **Progressive complexity**: Each lab builds on previous concepts
- **Learn by doing**: Every concept has hands-on exercises
- **Test-driven**: Validate your understanding with comprehensive tests
- **Real implementations**: You'll write actual working code, not pseudocode
- **Visual feedback**: Computation graph visualization for immediate validation

## 🛠️ Complete Solutions

Reference implementations are provided in:
- `engine.py` - Complete autograd Value class
- `nn.py` - Neural network building blocks (Neuron, Layer, MLP)
- `utils.py` - Topological sort and graph visualization
- Each notebook's "Correct Answer" sections

## 📊 What You'll Build

By the end of this course, you'll have:
1. ✅ Custom numeric computation library (understand NumPy internals)
2. ✅ Computation graph system (understand autograd)
3. ✅ Complete neural network from scratch (Neuron, Layer, MLP)
4. ✅ Trained model on 2D moon dataset (micrograd demo)
5. ✅ PyTorch MLP trained on MNIST with >95% accuracy
6. ✅ Transfer learning model on Kaggle Cats vs Dogs with >90% accuracy

## 🎯 Next Steps

After completing these labs, you'll be ready to:
- Read and understand PyTorch/TensorFlow source code
- Understand deep learning research papers
- Build custom neural network architectures
- Debug gradient flow issues
- Implement novel optimization algorithms

## 🙏 Acknowledgments

This lab series is inspired by **[Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd)** and his educational philosophy of building from first principles.

**Reference:**
- Original Repository: [github.com/karpathy/micrograd](https://github.com/karpathy/micrograd)
- Karpathy's YouTube Lecture: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Educational Approach: Learn by implementing automatic differentiation from scratch

This lab extends micrograd's concepts into a structured workshop format with:
- Progressive 5-lab curriculum
- Hands-on exercises with starter code
- Validation tests for immediate feedback
- Computation graph visualization using trace_graph from micrograd
- Complete reference implementations

**Micrograd License:** MIT License (Copyright (c) 2020 Andrej Karpathy)

## 📝 License

MIT License - Feel free to use for educational purposes!

---

**Happy Learning! 🚀**

Start with Lab 1 and work your way through. Take your time, run the code, experiment, and most importantly - have fun building your neural network from scratch!
