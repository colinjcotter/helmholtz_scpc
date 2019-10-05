from numpy import *
from scipy import optimize

def lsqr(z, *args):
    a, b = z
    epsilon = args[0]
    val = (epsilon**2*(4*a*b - epsilon**2*(a + b)**2) + (epsilon**2*(-a*b + 1) + 1)**2)/(epsilon**2*(a + 1)*(b + 1) + 1)**2
    print(val)
    return val

def grad_lsqr(z, *args):
    a, b = z
    epsilon = args[0]
    D1 = -2*epsilon**2*(b + 1)*(epsilon**2*(4*a*b - epsilon**2*(a + b)**2) + (epsilon**2*(-a*b + 1) + 1)**2)/(epsilon**2*(a + 1)*(b + 1) + 1)**3 + (-2*b*epsilon**2*(epsilon**2*(-a*b + 1) + 1) + epsilon**2*(4*b - epsilon**2*(2*a + 2*b)))/(epsilon**2*(a + 1)*(b + 1) + 1)**2
    D2 = -2*epsilon**2*(a + 1)*(epsilon**2*(4*a*b - epsilon**2*(a + b)**2) + (epsilon**2*(-a*b + 1) + 1)**2)/(epsilon**2*(a + 1)*(b + 1) + 1)**3 + (-2*a*epsilon**2*(epsilon**2*(-a*b + 1) + 1) + epsilon**2*(4*a - epsilon**2*(2*a + 2*b)))/(epsilon**2*(a + 1)*(b + 1) + 1)**2
    return array([D1, D2])

x0 = array([0.1,1.1])

result = optimize.minimize(lsqr, x0, args=(0.1), method="BFGS",
                           options={"gtol":1.0e-10,
                                    "maxiter":1000000}, jac=grad_lsqr)

print(result.x, result.fun**0.5, result.success, result.nit)
