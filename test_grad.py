from engine import Value
import torch

def test1():
    a = Value(3.0)
    b = Value(5.0)
    c = a*b + a
    c.backward()


    ta = torch.tensor(3.0, requires_grad=True)
    tb = torch.tensor(5.0, requires_grad=True)
    tc = ta*tb + ta
    tc.backward()

    print("a:", a.grad, ta.grad.item())
    print("b:", b.grad, tb.grad.item())



def test2():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a*b + c*a + b*b
    d.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(-3.0, requires_grad=True)
    tc = torch.tensor(10.0, requires_grad=True)
    td = ta*tb + tc*ta + tb*tb
    td.backward()

    print("a:", a.grad, ta.grad.item())
    print("b:", b.grad, tb.grad.item())
    print("c:", c.grad, tc.grad.item())

test1()
test2()



