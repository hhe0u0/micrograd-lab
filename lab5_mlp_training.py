# =============================================================================
# Lab 5: The "Robot Brain" — Training a Neural Network
# =============================================================================
#
# KEY CONCEPT — The Training Loop ("Loop of Life"):
#
#   ┌──────────────────────────────────────────────────────┐
#   │  1. FORWARD PASS    run model, get predictions       │
#   │  2. COMPUTE LOSS    measure how wrong we are (MSE)   │
#   │  3. ZERO GRAD       clear the "blame whiteboard"     │
#   │  4. BACKWARD PASS   assign blame to each weight      │
#   │  5. UPDATE WEIGHTS  nudge weights to reduce loss     │
#   │  (repeat many times)                                 │
#   └──────────────────────────────────────────────────────┘
#
# WHY ZERO GRAD?
#   Gradients accumulate with +=. Without zeroing, each training step
#   ADDS blame from ALL previous steps — the model gets confused.
#   zero_grad() is like erasing the whiteboard before each lesson.
#
# THE NEGATIVE STEP:
#   If loss.grad(weight) = +5, going UP on that weight makes the error bigger.
#   So we go the opposite way:  weight -= learning_rate * grad
#
# LEARNING RATE:
#   Too large  → model "jumps" over the solution and diverges.
#   Too small  → model improves but converges very slowly.
# =============================================================================

import random
import math

from engine import Value
from nn import MLP


# =============================================================================
# Dataset: Two concentric rings  (inner ring = +1, outer ring = −1)
# =============================================================================

random.seed(42)

def make_rings(n_per_class=15):
    """Generate a simple 2D binary classification dataset — no sklearn needed."""
    X, y = [], []
    for _ in range(n_per_class):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.3, 1.0)
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(1.0)
    for _ in range(n_per_class):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(1.5, 2.5)
        X.append([r * math.cos(angle), r * math.sin(angle)])
        y.append(-1.0)
    return X, y

X, y = make_rings(n_per_class=15)
print(f"Dataset: {len(X)} points  "
      f"({sum(1 for yi in y if yi > 0)} positive, "
      f"{sum(1 for yi in y if yi < 0)} negative)")


# =============================================================================
# Build the model
# =============================================================================
# MLP(2, [8, 8, 1]):
#   2 inputs → 8 ReLU neurons → 8 ReLU neurons → 1 linear output

model = MLP(2, [8, 8, 1])
print(f"Model: {model}")
print(f"Total parameters: {len(model.parameters())}\n")


# =============================================================================
# EXERCISE: Complete the five steps of the training loop
# =============================================================================
#
# Fill in each numbered TODO. After every 20 epochs the loss and accuracy
# are printed so you can watch the model improve.
#
# Expected behaviour:
#   Epoch   0 → loss ≈ 2–4  (random weights → bad predictions)
#   Epoch  60 → loss dropping, accuracy climbing
#   Epoch 200 → loss < 0.1, accuracy near 100 %
# =============================================================================

learning_rate = 0.05
n_epochs = 200

for epoch in range(n_epochs):

    # ── 1. FORWARD PASS ───────────────────────────────────────────────────────
    # Run the model on every input. Each output is a Value object.
    # TODO: ypred = [model(x) for x in X]
    ypred = None   # ← replace this line

    # ── 2. COMPUTE LOSS (Mean Squared Error) ──────────────────────────────────
    # MSE = average of (prediction − target)² across all samples
    # TODO: loss = sum((pred - yi)**2 for pred, yi in zip(ypred, y)) / len(y)
    loss = None    # ← replace this line

    # ── 3. ZERO GRADIENTS ─────────────────────────────────────────────────────
    # Clear accumulated gradients from the PREVIOUS step before computing new ones.
    # TODO: model.zero_grad()

    # ── 4. BACKWARD PASS ──────────────────────────────────────────────────────
    # Automatically compute gradients for every parameter.
    # TODO: loss.backward()

    # ── 5. UPDATE WEIGHTS (gradient descent) ──────────────────────────────────
    # Move each parameter in the direction that reduces the loss.
    # TODO: for p in model.parameters():
    #           p.data -= learning_rate * p.grad

    # ── Progress report ───────────────────────────────────────────────────────
    if epoch % 20 == 0:
        if loss is None or ypred is None:
            print(f"Epoch {epoch:3d} | (complete the TODOs above)")
        else:
            acc = sum(
                1 for p, t in zip(ypred, y)
                if (p.data > 0 and t > 0) or (p.data < 0 and t < 0)
            ) / len(y)
            print(f"Epoch {epoch:3d} | loss: {loss.data:.4f} | accuracy: {acc*100:.1f}%")


# =============================================================================
# Final evaluation
# =============================================================================

if ypred is not None:
    print("\nSample predictions (first 6 points):")
    for xi, yi, pred in zip(X[:6], y[:6], ypred[:6]):
        correct = 'CORRECT' if (pred.data > 0) == (yi > 0) else 'WRONG  '
        print(f"  {correct}  input={[f'{v:.2f}' for v in xi]}"
              f"  pred={pred.data:+.3f}  target={yi:+.1f}")

    # ── Optional: visualize the decision boundary (needs matplotlib) ──────────
    # try:
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #
    #     h = 0.05
    #     xx, yy = np.meshgrid(np.arange(-3, 3, h), np.arange(-3, 3, h))
    #     grid = [[xi, yi] for xi, yi in zip(xx.ravel(), yy.ravel())]
    #     Z = np.array([model(pt).data for pt in grid]).reshape(xx.shape)
    #
    #     plt.contourf(xx, yy, Z, levels=[-10, 0, 10], alpha=0.4, colors=['#ff8080', '#8080ff'])
    #     colors = ['red' if t > 0 else 'blue' for t in y]
    #     plt.scatter([xi[0] for xi in X], [xi[1] for xi in X], c=colors, edgecolors='k', s=40)
    #     plt.title("Decision Boundary after Training")
    #     plt.show()
    # except ImportError:
    #     print("(install matplotlib to visualize the decision boundary)")
