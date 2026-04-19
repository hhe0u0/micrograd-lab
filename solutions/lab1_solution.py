# Lab 1 Solution: Matrix class with magic methods

class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __add__(self, other):
        assert self.rows == other.rows and self.cols == other.cols, \
            f"Shape mismatch: ({self.rows},{self.cols}) vs ({other.rows},{other.cols})"
        result = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __mul__(self, other):
        assert self.cols == other.rows, \
            f"Shape mismatch: self.cols={self.cols} must equal other.rows={other.rows}"
        result = [
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def __repr__(self):
        rows_str = '\n  '.join(str(row) for row in self.data)
        return f"Matrix([\n  {rows_str}\n])"


if __name__ == '__main__':
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])

    C = A + B
    assert C.data == [[6, 8], [10, 12]]
    print("Addition: PASS")

    D = A * B
    assert D.data == [[19, 22], [43, 50]]
    print("Multiplication: PASS")

    E = Matrix([[1, 2, 3]])
    F = Matrix([[4], [5], [6]])
    G = E * F
    assert G.data == [[32]]
    print("Non-square matmul: PASS")

    print("All tests passed!")
