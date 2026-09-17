class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data 
        self.grad = 0
        self._prev = _children
        self._op = _op

    def __repr__(self):
        return f"Value=({self.data}), grad=({self.grad})"

    def __add__(self, other):
        out = self.data + other.data 
        out._prev = (self, other)
        out._op = ("+")

        def _backward():
            self.grad = 1*out.grad
            other.grad = 1*out.grad
            out._backward = _backward

        return out 

    def __mul__(self, other):
        out = self.data * other.data
        out._prev = (self, other)
        out._op = ("*")

        def _backward():
            self.grad = other.data*out.grad
            other.grad = self.grad*out.grad
            out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)

            build_topo(self)







        