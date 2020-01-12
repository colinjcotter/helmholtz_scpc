from firedrake import *
#A basic real-mode Helmholtz discretisation
#using a shifted preconditioner with LU

n = 100
mesh = UnitSquareMesh(n,n)

k = Constant(1.0)
ee = Constant(1.0)
ee0 = Constant(1.0)

deg = 1
BDMBroken = BrokenElement(FiniteElement("BDM", "triangle", deg))
V = VectorFunctionSpace(mesh, BDMBroken)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
Tr = VectorFunctionSpace(mesh, "HDiv Trace", deg)
W = MixedFunctionSpace((V,Q,Tr))

u, p, ll = TrialFunctions(W)
ur = u[0,:]
ui = u[1,:]
pr = p[0]
pi = p[1]
llr = ll[0]
lli = ll[1]

v, q, mu = TestFunctions(W)
vr = v[0,:]
vi = v[1,:]
qr = q[0]
qi = q[1]
mur = mu[0]
mui = mu[1]

r0 = Constant(1.0)

n = FacetNormal(mesh)

llrS = llr('+')
lliS = lli('+')
murS = mur('+')
muiS = mui('+')

a = (
    #ur equation
    (ee*inner(vr,ur) - k*inner(vr, ui) - r0*inner(div(vr), pr))*dx
    + r0*jump(vr,n)*llrS*dS + r0*inner(vr,n)*llr*ds
    #pr equation
    + (r0*inner(qr, div(ur)) - k*qr*pi + ee*qr*pr)*dx
    #mur equation
    + r0*jump(ur,n)*murS*dS + r0*inner(ur,n)*mur*ds
    #ui equation
    + (ee*inner(vi,ui) + k*inner(vi, ur) - r0*inner(div(vi), pi))*dx
    + r0*jump(vi,n)*lliS*dS + r0*inner(vi,n)*lli*ds
    #pi equation
    + (r0*inner(qi, div(ui)) + k*qi*pr + ee*qi*pi)*dx
    #mui equation
    + r0*jump(ui,n)*muiS*dS + r0*inner(ui,n)*mui*ds
    )

aP = (
    #ur equation
    (ee0*inner(vr,ur) - k*inner(vr, ui) - r0*inner(div(vr), pr))*dx
    + r0*jump(vr,n)*llrS*dS + r0*inner(vr,n)*llr*ds
    #pr equation
    + (r0*inner(qr, div(ur)) - k*qr*pi + ee0*qr*pr)*dx
    #mur equation
    + r0*jump(ur,n)*murS*dS + r0*inner(ur,n)*mur*ds
    #ui equation
    + (ee0*inner(vi,ui) + k*inner(vi, ur) - r0*inner(div(vi), pi))*dx
    + r0*jump(vi,n)*lliS*dS + r0*inner(vi,n)*lli*ds
    #pi equation
    + (r0*inner(qi, div(ui)) + k*qi*pr + ee0*qi*pi)*dx
    #mui equation
    + r0*jump(ui,n)*muiS*dS + r0*inner(ui,n)*mui*ds
    )

x, y = SpatialCoordinate(mesh)

f = exp(-((x-0.5)**2 + (y-0.5)**2)/0.25**2)

L = qi*f/k*dx

U = Function(W)

HProblem = LinearVariationalProblem(a, L, U, aP=aP, constant_jacobian=False)

lu_params = {'ksp_type':'gmres',
             'ksp_converged_reason':None,
             'mat_type':'aij',
             'ksp_rtol':1.0e-32,
             'ksp_atol':1.0e-6,
             'pc_type':'lu',
             'pc_factor_mat_solver_type':'mumps'}

condensed_params = {'ksp_type':'preonly',
                    'pc_type': 'gamg',
                    'pc_gamg_sym_graph': None,
                    'mg_levels': {'ksp_type': 'gmres',
                                  'ksp_max_it': 5,
                                  'pc_type': 'bjacobi',
                                  'sub_pc_type': 'ilu'}}

condensed_lu_params = {'ksp_type':'preonly',
                       'pc_type': 'lu',
                       'pc_factor_mat_solver_type':'mumps'}

hybrid_params = {'mat_type': 'matfree',
                 'pmat_type': 'matfree',
                 'ksp_type':'gmres',
                 'ksp_converged_reason':None,
                 'pc_type': 'python',
                 'pc_python_type':  'firedrake.SCPC',
                 'pc_sc_eliminate_fields': '0, 1',
                 'condensed_field':condensed_lu_params}

HSolver = LinearVariationalSolver(HProblem, solver_parameters=hybrid_params)

file0 = File('hh.pvd')

u_out = Function(V)
p_out = Function(Q)

for e0 in (1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001):
    ee.assign(e0)
    r0.assign(1.0)
    U.assign(0.)
    HSolver.solve()
    u, p, ll = U.split()
    u_out.assign(u)
    p_out.assign(p)
    ur = u_out.sub(0)
    ui = u_out.sub(0)
    pr = p_out.sub(0)
    pi = p_out.sub(0)
    file0.write(ur, ui, pr, pi)
