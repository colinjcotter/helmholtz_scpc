from firedrake import *
#A basic real-mode Helmholtz discretisation
#using a shifted preconditioner with LU

n = 100
mesh = UnitSquareMesh(50,50)

k = Constant(5)
ee = Constant(1.0)

V = VectorFunctionSpace(mesh, "DG", 1)
Q = FunctionSpace(mesh, "CG", 2)
W = MixedFunctionSpace((V,Q,V,Q))

ur, pr, ui, pi = TrialFunctions(W)
vr, qr, vi, qi = TestFunctions(W)

a = (
    -k*inner(vr, ui) + inner(vr, grad(pr))
    -inner(grad(qr), ur) -k*qr*pi
    +k*inner(vi, ur) + inner(vi, grad(pi))
    -inner(grad(qi), ui) +k*qi*pr
    )*dx

aP = (
    ee*inner(vr,ur) -k*inner(vr, ui) + inner(vr, grad(pr))
    -inner(grad(qr), ur) -k*qr*pi + ee*qr*pr
    +ee*inner(vi,ui) +k*inner(vi, ur) + inner(vi, grad(pi))
    -inner(grad(qi), ui) +k*qi*pr + ee*qi*pi
    )*dx

x, y = SpatialCoordinate(mesh)

f = exp(-((x-0.5)**2 + (y-0.5)**2)/0.25**2)

L = qi*f/k*dx

U = Function(W)

bcs = [DirichletBC(W.sub(1), 0., (1,2,3,4)),
       DirichletBC(W.sub(3), 0., (1,2,3,4))]

HProblem = LinearVariationalProblem(a, L, U, bcs=bcs, aP=aP,
                                    constant_jacobian=False)
params = {'ksp_type':'gmres',
          'ksp_converged_reason':None,
          'ksp_monitor_true_residual':None,
          'mat_type':'aij',
          'ksp_rtol':1.0e-32,
          'ksp_atol':1.0e-6,
          'pc_type':'lu',
          'pc_factor_mat_solver_type':'mumps'}
HSolver = LinearVariationalSolver(HProblem, solver_parameters=params)

for k0 in (1,10,100,100,1000,10000,100000):
    k.assign(k0)
    ee.assign(1.0)
    U.assign(0.)
    HSolver.solve()
