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

HProblem = LinearVariationalProblem(a, L, U, bcs=bcs, aP=aP)
params = {'ksp_type':'gmres',
          'ksp_converged_reason':None,
          'mat_type':'aij',
          'pc_type':'lu',
          'pc_factor_mat_solver_type':'mumps'}
HSolver = LinearVariationalSolver(HProblem, solver_parameters=params)

for k in range(10,100,10):
    print(k)
    U.assign(0.)
    HSolver.solve()
