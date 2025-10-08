import torch
from longshot import VAR_factory, XOR, avgQ

n = 10
VAR = VAR_factory(n)
circuit = XOR(*[VAR(i) for i in range(n)])
print(f"Circuit: {circuit.table}")
q = avgQ(circuit)
print(f"avgQ of XOR: {q}")
