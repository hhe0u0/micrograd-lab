def topological_sort(root):
    """Return nodes in topological order: output node first, leaf inputs last."""
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

    def _trace(root):
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
        """Visualize the computation graph. Requires: pip install graphviz"""
        nodes, edges = _trace(root)
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})
        for n in nodes:
            lbl = getattr(n, 'label', '')
            dot.node(
                name=str(id(n)),
                label=f"{{ {lbl} | data {n.data:.4f} | grad {n.grad:.4f} }}",
                shape='record',
            )
            if n._op:
                dot.node(name=str(id(n)) + n._op, label=n._op)
                dot.edge(str(id(n)) + n._op, str(id(n)))
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        return dot

except ImportError:
    def draw_dot(*args, **kwargs):
        print("graphviz not installed. To enable visualization:")
        print("  pip install graphviz")
        print("  brew install graphviz    # macOS")
        print("  apt-get install graphviz # Linux")
