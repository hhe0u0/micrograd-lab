# Lab 2 Solution: Value class — forward pass only

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), '*')

    def __pow__(self, other):   # bonus
        assert isinstance(other, (int, float))
        return Value(self.data ** other, (self,), f'**{other}')

    def __radd__(self, other):     return self + other
    def __rmul__(self, other):     return self * other
    def __neg__(self):             return self * -1
    def __sub__(self, other):      return self + (-other)
    def __rsub__(self, other):     return other + (-self)
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __repr__(self):
        lbl = f"'{self.label}' " if self.label else ''
        return f"Value({lbl}data={self.data:.4f}, grad={self.grad:.4f})"


if __name__ == '__main__':
    a = Value(2.0,  label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b;  e.label = 'e'
    d = e + c;  d.label = 'd'

    assert d.data == 4.0
    assert e in d._prev and c in d._prev
    assert a in e._prev and b in e._prev
    assert d._op == '+' and e._op == '*'
    print("All checks passed!")

    # Bonus: __pow__
    x = Value(3.0)
    y = x ** 2
    assert y.data == 9.0
    print("Bonus __pow__: PASS")
