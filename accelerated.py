from sympy import *

e, l, a = symbols("epsilon lambda alpha")

N = Matrix([[e, 0], [0, e]])
S = Matrix([[I*l, -1], [1, I*l]])
P = Matrix([[e**-1, 0], [0, e]])

Chat = (P-N)*(P+N).adjugate()

evals0 = list(Chat.eigenvals().keys())

l0 = evals0[0]/det(P+N)
l1 = evals0[1]/det(P+N)
