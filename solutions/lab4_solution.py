# Lab 4 Solution: Full Value class with automatic backpropagation

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils import topological_sort


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        nodes = topological_sort(self)
        self.grad = 1.0
        for node in nodes:
            node._backward()

    def __neg__(self):             return self * -1
    def __radd__(self, other):     return self + other
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __rmul__(self, other):     return self * other
    def __truediv__(self, other):  return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == '__main__':
    print("Running tests...")

    a = Value(3.0); b = Value(4.0)
    (a + b).backward()
    assert a.grad == 1.0 and b.grad == 1.0
    print("Test 1 (addition):          PASS")

    a = Value(3.0); b = Value(4.0)
    (a * b).backward()
    assert a.grad == 4.0 and b.grad == 3.0
    print("Test 2 (multiplication):    PASS")

    a = Value(3.0)
    (a + a).backward()
    assert a.grad == 2.0
    print("Test 3 (multivariate +=):   PASS")

    a = Value(2.0); b = Value(-3.0); c = Value(10.0)
    ((a * b) + c).backward()
    assert a.grad == -3.0 and b.grad == 2.0 and c.grad == 1.0
    print("Test 4 (chain):             PASS")

    a = Value(0.5); a.tanh().backward()
    h = 1e-5
    num = (math.tanh(0.5 + h) - math.tanh(0.5 - h)) / (2 * h)
    assert abs(a.grad - num) < 1e-4
    print("Test 5 (tanh):              PASS")

    a = Value(2.0);  a.relu().backward(); assert a.grad == 1.0
    a = Value(-2.0); a.relu().backward(); assert a.grad == 0.0
    print("Test 6 (relu):              PASS")

    print("\nAll tests passed!")
