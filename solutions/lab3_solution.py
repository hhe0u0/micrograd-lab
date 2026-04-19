# Lab 3 Solution: Manual backprop on  L = ((a * b) + c) * d

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data  = float(data)
        self.grad  = 0.0
        self._prev = set(_children)
        self._op   = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), '*')

    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self):         return self * -1
    def __sub__(self, other):  return self + (-other)

    def __repr__(self):
        lbl = f"'{self.label}' " if self.label else ''
        return f"Value({lbl}data={self.data:.4f}, grad={self.grad:.4f})"


# Build the expression: L = ((a * b) + c) * d
a = Value(2.0,  label='a')
b = Value(3.0,  label='b')
c = Value(1.0,  label='c')
d = Value(-1.0, label='d')

P = a * b;  P.label = 'P'   # P = 6
Q = P + c;  Q.label = 'Q'   # Q = 7
L = Q * d;  L.label = 'L'   # L = -7

print(f"L.data = {L.data}   (expected -7.0)")

# Manual backprop — working right to left:
L.grad = 1.0
Q.grad = d.data           # dL/dQ = d  = -1
d.grad = Q.data           # dL/dd = Q  =  7
P.grad = Q.grad * 1.0     # chain: dL/dP = dL/dQ × dQ/dP = -1 × 1 = -1
c.grad = Q.grad * 1.0     # chain: dL/dc = dL/dQ × dQ/dc = -1 × 1 = -1
a.grad = P.grad * b.data  # chain: dL/da = dL/dP × dP/da = -1 × 3 = -3
b.grad = P.grad * a.data  # chain: dL/db = dL/dP × dP/db = -1 × 2 = -2

print(f"\na.grad = {a.grad}   (expected -3.0)")
print(f"b.grad = {b.grad}   (expected -2.0)")
print(f"c.grad = {c.grad}   (expected -1.0)")
print(f"d.grad = {d.grad}   (expected  7.0)")

# Numerical gradient verification
def numerical_grad(val, h=1e-4):
    original = val.data

    val.data = original + h
    L_plus = ((Value(a.data) * Value(b.data)) + Value(c.data)) * Value(d.data)

    val.data = original - h
    L_minus = ((Value(a.data) * Value(b.data)) + Value(c.data)) * Value(d.data)

    val.data = original
    return (L_plus.data - L_minus.data) / (2 * h)

print("\nNumerical verification:")
print(f"  a: analytical={a.grad:.4f}  numerical={numerical_grad(a):.4f}")
print(f"  b: analytical={b.grad:.4f}  numerical={numerical_grad(b):.4f}")
print(f"  c: analytical={c.grad:.4f}  numerical={numerical_grad(c):.4f}")
print(f"  d: analytical={d.grad:.4f}  numerical={numerical_grad(d):.4f}")
