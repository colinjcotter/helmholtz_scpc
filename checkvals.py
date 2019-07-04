from scipy import *
from scipy.linalg import inv, eig

ee = 0.001

lrmax = 0.
limax = 0.

for s in arange(1000):
    P = matrix([[1+ee,0],[1,1]])
    S = matrix([[1j*s,0],[0,1j*s]])
    N = matrix([[ee,-1],[1,ee]])

    C = (P-N)*inv(P+N)*(P+S)*inv(P-S)

    e, v = eig(C)

    lrmax = max(lrmax, abs(e[0].real))
    lrmax = max(lrmax, abs(e[1].real))
    limax = max(limax, abs(e[0].imag))
    limax = max(limax, abs(e[1].imag))

print(lrmax, limax)
a = lrmax
b = limax
print(((a+b)/(1+sqrt(1+b**2-a**2)))**0.5)
