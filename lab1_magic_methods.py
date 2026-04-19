# =============================================================================
# Lab 1: Python "Superpowers" — Magic Methods
# =============================================================================
#
# Python's "magic methods" (also called dunder methods) let you define what
# operators do when applied to your custom classes. This is the foundation of
# micrograd: a Value object behaves just like a number, but secretly records
# every operation it's involved in.
#
# Examples:
#   __add__(self, other)   → called when you write   self + other
#   __mul__(self, other)   → called when you write   self * other
#   __repr__(self)         → called when you  print  the object
#
# KEY INSIGHT: Even built-in numbers are objects — int.__add__ is the magic
# method that defines what 3 + 5 means. micrograd gives us smarter numbers
# that remember their history.
# =============================================================================


# =============================================================================
# DEMO: A class that greets you whenever two instances are added
# =============================================================================

class Greeter:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        print(f"Hello from {self.name} and {other.name}!")
        return Greeter(f"{self.name}+{other.name}")

    def __repr__(self):
        return f"Greeter('{self.name}')"


# Try it — Python calls a.__add__(b) behind the scenes:
a = Greeter("Alice")
b = Greeter("Bob")
c = a + b
print(f"Result: {c}")
print()


# =============================================================================
# EXERCISE: Implement a Matrix class
# =============================================================================
#
# A Matrix wraps a 2D list and supports:
#   A + B  →  element-wise addition   (shapes must match)
#   A * B  →  matrix multiplication   (A.cols must equal B.rows)
#
# Worked example:
#   A = [[1, 2],      B = [[5, 6],
#        [3, 4]]           [7, 8]]
#
#   A + B = [[6, 8], [10, 12]]
#
#   A * B:  row 0 of A · col 0 of B = 1×5 + 2×7 = 19
#           row 0 of A · col 1 of B = 1×6 + 2×8 = 22
#           row 1 of A · col 0 of B = 3×5 + 4×7 = 43
#           row 1 of A · col 1 of B = 3×6 + 4×8 = 50
#   A * B = [[19, 22], [43, 50]]
# =============================================================================

class Matrix:
    def __init__(self, data):
        # TODO: Store self.data = data
        # TODO: self.rows = len(data)
        # TODO: self.cols = len(data[0])
        raise NotImplementedError("Implement __init__")

    def __add__(self, other):
        # TODO: assert self.rows == other.rows and self.cols == other.cols
        # TODO: result[i][j] = self.data[i][j] + other.data[i][j]
        # TODO: return Matrix(result)
        raise NotImplementedError("Implement __add__")

    def __mul__(self, other):
        # TODO: assert self.cols == other.rows
        # TODO: result[i][j] = sum(self.data[i][k] * other.data[k][j]
        #                          for k in range(self.cols))
        #       for i in range(self.rows), j in range(other.cols)
        # TODO: return Matrix(result)
        raise NotImplementedError("Implement __mul__")

    def __repr__(self):
        # TODO: return a readable multi-row string
        # Hint: '\n'.join(str(row) for row in self.data)
        raise NotImplementedError("Implement __repr__")


# =============================================================================
# Tests — run this file to check your work
# =============================================================================

if __name__ == '__main__':
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    print("A =\n", A)
    print("B =\n", B)

    C = A + B
    print("\nA + B =\n", C)
    assert C.data == [[6, 8], [10, 12]], f"Addition failed: {C.data}"
    print("  Addition: PASS")

    D = A * B
    print("\nA * B =\n", D)
    assert D.data == [[19, 22], [43, 50]], f"Multiplication failed: {D.data}"
    print("  Multiplication: PASS")

    # Bonus: non-square matrices
    E = Matrix([[1, 2, 3]])      # shape 1×3
    F = Matrix([[4], [5], [6]])  # shape 3×1
    G = E * F
    assert G.data == [[32]], f"Non-square matmul failed: {G.data}"
    print("  Non-square matmul: PASS")

    print("\nAll tests passed!")
