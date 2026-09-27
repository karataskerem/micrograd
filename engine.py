class Value:
    def __init__(self, data, _prev=(), _op=""):
        self.data = data 
        self.grad = 0
        self._prev = _prev
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value=({self.data}), grad=({self.grad})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self,other), "+") 
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward

        return out 

    def __mul__(self, other):
        out = Value(self.data*other.data, (self,other), "*")
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
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
        self.grad = 1.0           

        for node in reversed(topo):   
            node._backward()

            



            


