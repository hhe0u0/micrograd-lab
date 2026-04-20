"""
Utility functions for micrograd labs.

Includes:
- Topological sort for backpropagation
- Computation graph visualization (from Karpathy's micrograd)

Visualization code adapted from:
https://github.com/karpathy/micrograd/blob/master/trace_graph.ipynb

Original micrograd by Andrej Karpathy (MIT License, 2020)
"""


def topological_sort(root):
    """
    Return nodes in topological order: output node first, leaf inputs last.

    This is used in Lab 4 for implementing automatic backpropagation.

    Args:
        root: The output Value node to start from

    Returns:
        List of nodes in topological order (reversed)
    """
    topo = []
    visited = set()

    def visit(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                visit(child)
            topo.append(node)

    visit(root)
    return list(reversed(topo))


try:
    from graphviz import Digraph

    def trace(root):
        """
        Build a set of all nodes and edges in the computation graph.

        This function traverses the computation graph backwards from the
        output node to collect all nodes and edges.

        Args:
            root: The output Value node to trace backwards from

        Returns:
            nodes: Set of all Value nodes in the graph
            edges: Set of (child, parent) tuples representing edges
        """
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)

        build(root)
        return nodes, edges

    def draw_dot(root, format='svg', rankdir='LR'):
        """
        Draw the computation graph starting from root Value node.

        Visualizes the forward pass (data values) and backward pass (gradients)
        in a directed graph. Each node shows data and grad values.

        Args:
            root: The output Value node to visualize
            format: Output format ('svg', 'png', etc.)
            rankdir: Graph direction ('LR' = left-to-right, 'TB' = top-to-bottom)

        Returns:
            A Graphviz Digraph object that renders in Jupyter notebooks

        Example:
            >>> from engine import Value
            >>> x = Value(2.0, label='x')
            >>> y = Value(3.0, label='y')
            >>> z = x * y; z.label = 'z'
            >>> z.backward()
            >>> draw_dot(z)  # Displays computation graph with gradients
        """
        assert rankdir in ['LR', 'TB'], "rankdir must be 'LR' or 'TB'"

        nodes, edges = trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

        # Add nodes to the graph
        for n in nodes:
            # Get label if it exists, otherwise use empty string
            label = getattr(n, 'label', '')
            # Create node showing: label | data | gradient
            node_label = "{ %s | data %.4f | grad %.4f }" % (label, n.data, n.grad)
            dot.node(name=str(id(n)), label=node_label, shape='record')

            # Add operation node if this value was produced by an operation
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))

        # Add edges connecting nodes
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot

    def visualize_graph(root, title=None):
        """
        Convenience function to visualize a computation graph with optional title.

        Useful for Lab 2 and Lab 3 to validate computation graphs after
        forward pass and backward pass.

        Args:
            root: The output Value node to visualize
            title: Optional title to print above the graph

        Returns:
            The Graphviz Digraph object

        Example:
            >>> a = Value(2.0, label='a')
            >>> b = Value(-3.0, label='b')
            >>> c = a * b; c.label = 'c'
            >>> c.backward()
            >>> visualize_graph(c, "Multiplication: forward and backward")
        """
        if title:
            print(f"\n{'='*60}")
            print(f" {title}")
            print(f"{'='*60}\n")

        return draw_dot(root)

except ImportError:
    def trace(*args, **kwargs):
        print("⚠️  graphviz not installed. Computation graph visualization unavailable.")
        print("To enable visualization:")
        print("  pip install graphviz")
        print("  brew install graphviz    # macOS")
        print("  apt-get install graphviz # Linux")
        return set(), set()

    def draw_dot(*args, **kwargs):
        print("⚠️  graphviz not installed. Cannot draw computation graph.")
        print("To enable visualization:")
        print("  pip install graphviz")
        print("  brew install graphviz    # macOS")
        print("  apt-get install graphviz # Linux")

    def visualize_graph(*args, **kwargs):
        print("⚠️  graphviz not installed. Cannot visualize computation graph.")
        print("To enable visualization:")
        print("  pip install graphviz")
        print("  brew install graphviz    # macOS")
        print("  apt-get install graphviz # Linux")
