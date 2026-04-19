# =============================================================================
# Lab 3: The Chain Rule — Manual Backpropagation
# =============================================================================
#
# KEY CONCEPT — The Chain Rule:
#   If f depends on e, and e depends on a, then:
#       df/da = (df/de) × (de/da)
#
#   This "chain" of multiplications traces how ANY input affects the final
#   output, no matter how deeply nested the computation is.
#
# ANALOGY — Car vs. Bicycle vs. Walking:
#   • Walking speed:   1 km/h
#   • Cycling:        10× faster than walking  →  10 km/h
#   • Driving:        10× faster than cycling  →  100 km/h
#   Car speed = (cycling/walking) × (driving/cycling) × walking = 10 × 10 × 1
#   The chain rule multiplies these "speedups" (sensitivities) together.
#
# In neural networks, "gradient" means "sensitivity":
#   If a.grad = 10, then nudging `a` by +0.001 changes the output by ≈ +0.010.
# =============================================================================


# ── A complete forward-only Value class is provided for this lab ─────────────

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


# =============================================================================
# DEMO: Manual backprop on  f = (a + b) * c
# =============================================================================
#
# Forward pass:
#   e = a + b  =  2 + (-3)  =  -1
#   f = e * c  =  (-1) * 10 =  -10
#
# Backward pass (work right to left):
#   df/df = 1                                 (by definition)
#   df/de = c.data  =  10                     (d(e*c)/de = c)
#   df/dc = e.data  =  -1                     (d(e*c)/dc = e)
#   df/da = df/de × de/da  =  10 × 1  =  10  (chain rule; de/da = 1 since e = a+b)
#   df/db = df/de × de/db  =  10 × 1  =  10  (chain rule; de/db = 1)
# =============================================================================

print("=" * 55)
print("DEMO: f = (a + b) * c")
print("=" * 55)

a = Value(2.0,  label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')

e = a + b;  e.label = 'e'
f = e * c;  f.label = 'f'

# Set gradients manually:
f.grad = 1.0
c.grad = e.data           # df/dc = e
e.grad = c.data           # df/de = c
a.grad = e.grad * 1.0     # chain rule: df/da = df/de × de/da = 10 × 1
b.grad = e.grad * 1.0     # chain rule: df/db = df/de × de/db = 10 × 1

print(f"  a.grad = {a.grad}   (expected  10.0)")
print(f"  b.grad = {b.grad}   (expected  10.0)")
print(f"  c.grad = {c.grad}   (expected  -1.0)")

# Sanity-check with a numerical gradient (bump a by h and see how f changes):
h = 1e-4
f_plus = (Value(a.data + h) + Value(b.data)) * Value(c.data)
numerical_a = (f_plus.data - f.data) / h
print(f"\n  Numerical df/da ≈ {numerical_a:.4f}  (should match a.grad = {a.grad})")
print()


# =============================================================================
# EXERCISE: Manual backprop on  L = ((a * b) + c) * d
# =============================================================================
#
# Given: a=2, b=3, c=1, d=-1
#
# Step 1 — Build the expression using Value objects:
#   P = a * b
#   Q = P + c
#   L = Q * d
#   Verify: L.data should be -7.0
#
# Step 2 — Set gradients by working backwards from L:
#   L.grad = 1.0
#   Then figure out Q.grad, d.grad, P.grad, c.grad, a.grad, b.grad
#
# HINTS:
#   For L = Q * d:  dL/dQ = d.data,  dL/dd = Q.data
#   For Q = P + c:  dQ/dP = 1,       dQ/dc = 1
#   For P = a * b:  dP/da = b.data,  dP/db = a.data
#   Chain rule:  a.grad = (dL/dQ) × (dQ/dP) × (dP/da)
#
# Step 3 — Verify using the numerical checker below (uncomment when ready).
# =============================================================================

print("=" * 55)
print("EXERCISE: L = ((a * b) + c) * d")
print("=" * 55)

a = Value(2.0,  label='a')
b = Value(3.0,  label='b')
c = Value(1.0,  label='c')
d = Value(-1.0, label='d')

# TODO: Build the expression
# P = a * b;  P.label = 'P'
# Q = P + c;  Q.label = 'Q'
# L = Q * d;  L.label = 'L'
# print(f"L.data = {L.data}   (expected -7.0)")

# TODO: Set gradients manually, working backwards from L
# L.grad = 1.0
# ...

# ── Numerical gradient checker (uncomment after setting gradients) ────────────
def numerical_grad(val, h=1e-4):
    """Estimate dL/d(val) by rebuilding L after bumping val.data by ±h."""
    original = val.data

    val.data = original + h
    _a, _b, _c, _d = Value(a.data), Value(b.data), Value(c.data), Value(d.data)
    L_plus = ((_a * _b) + _c) * _d

    val.data = original - h
    _a, _b, _c, _d = Value(a.data), Value(b.data), Value(c.data), Value(d.data)
    L_minus = ((_a * _b) + _c) * _d

    val.data = original
    return (L_plus.data - L_minus.data) / (2 * h)

# TODO: After filling in gradients, uncomment these lines to verify:
# print(f"\n  a: analytical = {a.grad:.4f}  |  numerical = {numerical_grad(a):.4f}")
# print(f"  b: analytical = {b.grad:.4f}  |  numerical = {numerical_grad(b):.4f}")
# print(f"  c: analytical = {c.grad:.4f}  |  numerical = {numerical_grad(c):.4f}")
# print(f"  d: analytical = {d.grad:.4f}  |  numerical = {numerical_grad(d):.4f}")
