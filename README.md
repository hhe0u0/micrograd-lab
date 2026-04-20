# Micrograd Lab - Interactive Neural Network Course

A hands-on Jupyter notebook series that teaches you how to build a neural network from scratch, step by step.

## 🎯 What You'll Learn

Build your own automatic differentiation engine and neural network library, understanding the core principles behind modern deep learning frameworks like PyTorch and TensorFlow.

## 📚 Lab Structure

### Lab 1: Python Magic Methods
**Concepts:** Operator overloading, custom classes, magic methods
**Demo:** Chinese family relation calculator (Mother() + Sister(younger=True) = 小姨)
**Exercise:** Implement Matrix class with addition and multiplication

### Lab 2: The Value Object & Forward Pass
**Concepts:** Computation graphs, forward propagation, tracking operations
**Exercise:** Build the Value class that records data and parent relationships

### Lab 3: Manual Backpropagation with Chain Rule
**Concepts:** Chain rule, gradients, sensitivity analysis
**Exercise:** Manually compute gradients for nested expressions

### Lab 4: Automatic Backpropagation Engine
**Concepts:** Topological sort, _backward closures, gradient accumulation
**Exercise:** Implement automatic differentiation (autograd)

### Lab 5: Training a Neural Network
**Concepts:** Training loop, MSE loss, gradient descent, learning rate
**Exercise:** Train a 2-layer MLP on concentric rings dataset

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Installation

```bash
# Install Jupyter if not already installed
pip install notebook

# Clone or download this repository
cd micrograd-lab

# Install dependencies
pip install -r requirements.txt

# Start Jupyter
jupyter notebook
```

### Running Labs

Open each lab notebook in order:
1. `lab1_magic_methods.ipynb`
2. `lab2_value_forward.ipynb`
3. `lab3_chain_rule.ipynb`
4. `lab4_auto_backprop.ipynb`
5. `lab5_mlp_training.ipynb`

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
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── lab1_magic_methods.ipynb      # Lab 1 notebook
├── lab2_value_forward.ipynb      # Lab 2 notebook
├── lab3_chain_rule.ipynb         # Lab 3 notebook
├── lab4_auto_backprop.ipynb      # Lab 4 notebook
├── lab5_mlp_training.ipynb       # Lab 5 notebook
├── engine.py                      # Complete Value class (reference)
├── nn.py                          # Neural network layers (reference)
├── utils.py                       # Helper functions (topological sort, visualization)
└── family_relations.py            # Chinese family relation calculator demo
```

## 🎓 Learning Philosophy

- **Progressive complexity**: Each lab builds on previous concepts
- **Learn by doing**: Every concept has hands-on exercises
- **Test-driven**: Validate your understanding with comprehensive tests
- **Real implementations**: You'll write actual working code, not pseudocode
- **Cultural relevance**: Uses engaging examples like Chinese family relations

## 🛠️ Complete Solutions

Reference implementations are provided in:
- `engine.py` - Complete autograd Value class
- `nn.py` - Neural network building blocks (Neuron, Layer, MLP)
- `utils.py` - Topological sort and graph visualization
- Each notebook's "Correct Answer" sections

## 📊 What You'll Build

By the end of this course, you'll have:
1. ✅ A working autograd engine (automatic differentiation)
2. ✅ Neural network layers (neurons, layers, MLP)
3. ✅ Training loop implementation
4. ✅ A trained model that classifies 2D data

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
- Cultural examples and edge cases
- Complete reference implementations

**Micrograd License:** MIT License (Copyright (c) 2020 Andrej Karpathy)

## 📝 License

MIT License - Feel free to use for educational purposes!

---

**Happy Learning! 🚀**

Start with Lab 1 and work your way through. Take your time, run the code, experiment, and most importantly - have fun building your neural network from scratch!
