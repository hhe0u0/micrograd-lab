# Lab 5 Solution: Complete training loop

import random
import math

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engine import Value
from nn import MLP

random.seed(42)


def make_rings(n_per_class=15):
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
model = MLP(2, [8, 8, 1])

learning_rate = 0.05
n_epochs = 200

for epoch in range(n_epochs):

    # 1. Forward pass
    ypred = [model(x) for x in X]

    # 2. Compute MSE loss
    loss = sum((pred - yi)**2 for pred, yi in zip(ypred, y)) / len(y)

    # 3. Zero gradients
    model.zero_grad()

    # 4. Backward pass
    loss.backward()

    # 5. Update weights
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 20 == 0:
        acc = sum(
            1 for p, t in zip(ypred, y)
            if (p.data > 0 and t > 0) or (p.data < 0 and t < 0)
        ) / len(y)
        print(f"Epoch {epoch:3d} | loss: {loss.data:.4f} | accuracy: {acc*100:.1f}%")

print("\nFinal predictions (first 6 points):")
for xi, yi, pred in zip(X[:6], y[:6], ypred[:6]):
    correct = 'CORRECT' if (pred.data > 0) == (yi > 0) else 'WRONG  '
    print(f"  {correct}  input={[f'{v:.2f}' for v in xi]}"
          f"  pred={pred.data:+.3f}  target={yi:+.1f}")
