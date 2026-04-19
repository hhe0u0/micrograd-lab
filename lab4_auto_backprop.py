# =============================================================================
# Lab 4: Building the Engine — Automatic Backpropagation
# =============================================================================
#
# In Lab 3 we set gradients BY HAND. That works for tiny expressions but is
# impossible for a network with thousands of operations.
#
# The trick: when each Value is created, store a _backward() closure that
# knows exactly how to pass gradients to its parents for that one operation.
# Then visit every node in the right order and call _backward().
#
# TOPOLOGICAL SORT — the right visiting order:
#   Like following a recipe in reverse — you can't un-bake a cake until you
#   know its ingredients. topological_sort(root) (from utils.py) visits every
#   node from the "finished result" back to the "raw inputs."
#
# CRITICAL — always use  +=  when accumulating gradients, never  =
#   If you write  x = a + a, then 'a' is a parent TWICE.
#   Each path must contribute separately:
#       a.grad += 1 * out.grad   ← from left  'a'
#       a.grad += 1 * out.grad   ← from right 'a'
#   Using  a.grad =  would overwrite the first contribution (the "multivariate
#   bug"). This is the single most common mistake — Test 3 below catches it.
# =============================================================================

import math
from utils import topological_sort


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ── EXERCISE: fill in each _backward closure ──────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Derivative of (a + b) w.r.t. a is 1; w.r.t. b is 1.
            # TODO: self.grad  += 1.0 * out.grad
            # TODO: other.grad += 1.0 * out.grad
            pass

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Derivative of (a * b) w.r.t. a is b; w.r.t. b is a.
            # TODO: self.grad  += other.data * out.grad
            # TODO: other.grad += self.data  * out.grad
            pass

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Derivative of x^n w.r.t. x is  n * x^(n-1).
            # TODO: self.grad += other * (self.data ** (other - 1)) * out.grad
            pass

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # Derivative of tanh(x) w.r.t. x is  1 - tanh(x)^2  =  1 - t^2.
            # TODO: self.grad += (1 - t ** 2) * out.grad
            pass

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            # ReLU passes the gradient through only when the output is positive.
            # TODO: self.grad += (out.data > 0) * out.grad
            pass

        out._backward = _backward
        return out

    def backward(self):
        # ── EXERCISE: implement automatic backpropagation ────────────────────
        #
        # 1. Get all nodes in topological order (output first, inputs last):
        #       nodes = topological_sort(self)
        #
        # 2. Seed the output gradient:
        #       self.grad = 1.0
        #
        # 3. Call _backward() on every node in order:
        #       for node in nodes:
        #           node._backward()
        #
        # TODO: implement this
        pass

    # ── Pre-implemented helpers ───────────────────────────────────────────────
    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# =============================================================================
# Tests — run this file after implementing the above
# =============================================================================

if __name__ == '__main__':
    print("Running tests...\n")

    # Test 1: Addition backward
    a = Value(3.0); b = Value(4.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0 and b.grad == 1.0, \
        f"Test 1 FAIL: a.grad={a.grad}, b.grad={b.grad}"
    print("Test 1 (addition backward):          PASS")

    # Test 2: Multiplication backward
    a = Value(3.0); b = Value(4.0)
    c = a * b
    c.backward()
    assert a.grad == 4.0 and b.grad == 3.0, \
        f"Test 2 FAIL: a.grad={a.grad}, b.grad={b.grad}"
    print("Test 2 (multiplication backward):    PASS")

    # Test 3: Multivariate — same variable used twice (catches  =  instead of  +=)
    a = Value(3.0)
    b = a + a
    b.backward()
    assert a.grad == 2.0, \
        f"Test 3 FAIL: a.grad={a.grad}  (expected 2.0 — did you use  +=  ?)"
    print("Test 3 (multivariate / += check):    PASS")

    # Test 4: Chained expression matching Lab 3 demo
    a = Value(2.0); b = Value(-3.0); c = Value(10.0)
    d = (a * b) + c
    d.backward()
    assert a.grad == -3.0, f"Test 4 FAIL: a.grad={a.grad}"
    assert b.grad ==  2.0, f"Test 4 FAIL: b.grad={b.grad}"
    assert c.grad ==  1.0, f"Test 4 FAIL: c.grad={c.grad}"
    print("Test 4 (chained expression):         PASS")

    # Test 5: tanh backward (numerical check)
    a = Value(0.5)
    b = a.tanh()
    b.backward()
    h = 1e-5
    numerical = (math.tanh(0.5 + h) - math.tanh(0.5 - h)) / (2 * h)
    assert abs(a.grad - numerical) < 1e-4, \
        f"Test 5 FAIL: a.grad={a.grad:.6f}, expected≈{numerical:.6f}"
    print("Test 5 (tanh backward):              PASS")

    # Test 6: ReLU backward
    a = Value(2.0);  b = a.relu();  b.backward()
    assert a.grad == 1.0, f"Test 6a FAIL: expected 1.0, got {a.grad}"
    a = Value(-2.0); b = a.relu();  b.backward()
    assert a.grad == 0.0, f"Test 6b FAIL: expected 0.0, got {a.grad}"
    print("Test 6 (relu backward):              PASS")

    print("\nAll tests passed! Your autograd engine is working.")
