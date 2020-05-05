from firedrake import *
#A basic real-mode Helmholtz discretisation
#using GTMG

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

x, y = SpatialCoordinate(mesh)

f = exp(-((x-0.5)**2 + (y-0.5)**2)/0.25**2)

L = qi*f/k*dx

U = Function(W)

HProblem = LinearVariationalProblem(a, L, U, constant_jacobian=False)

gtmg = {
    'ksp_type': 'gmres',
    'mat_type': 'matfree',
    'pc_type': 'python',
    'pc_python_type': 'firedrake.GTMGPC',
    'gt': {'mg_levels': {
        'ksp_type': 'chebyshev',
        'ksp_max_it': 3},
           'mg_coarse': {'ksp_type': 'preonly',
                         'pc_type': 'lu'}}}

hybrid_params = {'mat_type': 'matfree',
                 'pmat_type': 'matfree',
                 'ksp_type':'preonly',
                 'ksp_converged_reason':None,
                 'pc_type': 'python',
                 'pc_python_type':  'firedrake.SCPC',
                 'pc_sc_eliminate_fields': '0, 1',
                 'condensed_field':gtmg}

def get_p1_space():
    W = VectorFunctionSpace(mesh, "CG", 1)
    return W

#coarse operator notes (... indicates RHS stuff we don't care about)
# (e+ik)*u + r0*grad p = ...
# (e+ik)*p + r0*div u = ...
# eliminate u
#
# (e+ik)^2 p + r0*div (e+ik) u = ...
# (e+ik)^2 p - r0^2*div grad p = ...
# -k^2 + e^2 - 2ike

def p1_callback():
    W = get_p1_space()
    p = TrialFunction(W)
    pr = p[0]
    pi = p[1]
    q = TestFunction(W)
    qr = q[0]
    qi = q[1]
    eqnr = qr*((-k**2 + e**2)*pr + 2*e*k*pi) + r0**2*inner(grad(qr),grad(pr))
    eqni = qi*((-k**2 + e**2)*pi - 2*e*k*pr) + r0**2*inner(grad(qi),grad(pi))
    return eqnr + eqni

appctx = {'get_coarse_operator': p1_callback,
          'get_coarse_space': get_p1_space}

HSolver = LinearVariationalSolver(HProblem, solver_parameters=hybrid_params,
                                  appctx = appctx)
HSolver.solve()
