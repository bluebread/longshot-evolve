from longshot import VAR_factory, XOR, avgQ

VAR = VAR_factory(3)
x0, x1, x2 = VAR(0), VAR(1), VAR(2)
circuit = XOR(x0, x1, x2)
print(f"Circuit: {circuit.table}")
q = avgQ(circuit)
print(f"avgQ of XOR(x0, x1, x2): {q}")
