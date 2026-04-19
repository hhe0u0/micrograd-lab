# =============================================================================
# Lab 2: The Value Object & The Forward Pass
# =============================================================================
#
# Every computation in micrograd creates a "story" that records:
#   • What number was produced   (self.data)
#   • Which nodes were involved  (self._prev)  ← the "family tree"
#   • What operation was used    (self._op)
#
# This story is called the COMPUTATION GRAPH.
# When you write  c = a + b, Python calls a.__add__(b), which:
#   1. Computes    c.data  = a.data + b.data
#   2. Remembers   c._prev = {a, b}      ← "c was born from a and b"
#   3. Remembers   c._op   = '+'         ← "using addition"
#
# The computation graph is what makes automatic backprop possible (Lab 4).
# For now we only implement the forward direction — gradients come later.
# =============================================================================


# =============================================================================
# EXERCISE: Implement the Value class (forward pass only)
# =============================================================================

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        # TODO: self.data  = float(data)
        # TODO: self.grad  = 0.0             ← gradient starts at zero
        # TODO: self._prev = set(_children)  ← parents in the computation graph
        # TODO: self._op   = _op             ← operation that created this node
        # TODO: self.label = label            ← optional human-readable name
        # TODO: self._backward = lambda: None ← placeholder (used in Lab 4)
        raise NotImplementedError("Implement __init__")

    def __add__(self, other):
        # Wrap plain numbers so  val + 2  works as well as  val + Value(2)
        other = other if isinstance(other, Value) else Value(other)
        # TODO: return Value(self.data + other.data, (self, other), '+')
        raise NotImplementedError("Implement __add__")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # TODO: return Value(self.data * other.data, (self, other), '*')
        raise NotImplementedError("Implement __mul__")

    # ── Already implemented — these build on __add__ and __mul__ ─────────────
    def __radd__(self, other):     return self + other       # other + self
    def __rmul__(self, other):     return self * other       # other * self
    def __neg__(self):             return self * -1          # -self
    def __sub__(self, other):      return self + (-other)    # self - other
    def __rsub__(self, other):     return other + (-self)    # other - self
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __repr__(self):
        lbl = f"'{self.label}' " if self.label else ''
        return f"Value({lbl}data={self.data:.4f}, grad={self.grad:.4f})"


# =============================================================================
# Demo & Tests — run after filling in the class above
# =============================================================================

if __name__ == '__main__':
    from utils import draw_dot

    # Build the expression: d = (a * b) + c
    a = Value(2.0,  label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b;  e.label = 'e'   # e = 2 × (-3) = -6
    d = e + c;  d.label = 'd'   # d = (-6) + 10 = 4

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"e = a * b = {e}")
    print(f"d = e + c = {d}")

    # ── Verify the family tree ────────────────────────────────────────────────
    assert d.data == 4.0,  f"d.data should be 4.0, got {d.data}"
    assert e in d._prev,   "e should be in d._prev"
    assert c in d._prev,   "c should be in d._prev"
    assert a in e._prev,   "a should be in e._prev"
    assert b in e._prev,   "b should be in e._prev"
    print("\nFamily tree checks: PASS")

    print(f"\nd._op = '{d._op}'   (expected '+')")
    print(f"e._op = '{e._op}'   (expected '*')")

    # ── Visualize (optional — needs: pip install graphviz) ────────────────────
    # dot = draw_dot(d)
    # dot.render('lab2_graph', view=True)
    # print("Visualization saved to lab2_graph.svg")

    # ── Bonus: implement __pow__ ──────────────────────────────────────────────
    # Add this method to Value so that  val ** 2  works:
    #
    #   def __pow__(self, other):
    #       assert isinstance(other, (int, float))
    #       return Value(self.data ** other, (self,), f'**{other}')
    #
    # Then test:  x = Value(3.0);  y = x ** 2;  assert y.data == 9.0

    print("\nForward pass: PASS")
