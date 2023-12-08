from firedrake import *
#A basic real-mode Helmholtz discretisation
#using hybridisation with LU

n = 50
L = 1.0
m = IntervalMesh(n, L)
mesh = ExtrudedMesh(m, layers=n)

k = Constant(1.0)
ee = Constant(1.0)

deg = 1
RTBroken = BrokenElement(FiniteElement("RTCF", "quadrilateral", deg))
V = VectorFunctionSpace(mesh, RTBroken)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
Tr = VectorFunctionSpace(mesh, "HDiv Trace", deg-1)
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

n = FacetNormal(mesh)

llrS = llr('+')
lliS = lli('+')
murS = mur('+')
muiS = mui('+')

dS = dS_h + dS_v
ds = ds_tb + ds_v

a = (
    #ur equation
    (ee*inner(vr,ur) - k*inner(vr, ui) - inner(div(vr), pr))*dx
    + jump(vr,n)*llrS*dS + inner(vr,n)*llr*ds
    #pr equation
    + (inner(qr, div(ur)) - k*qr*pi + ee*qr*pr)*dx
    #mur equation
    + jump(ur,n)*murS*dS + inner(ur,n)*mur*ds
    #ui equation
    + (ee*inner(vi,ui) + k*inner(vi, ur) - inner(div(vi), pi))*dx
    + jump(vi,n)*lliS*dS + inner(vi,n)*lli*ds
    #pi equation
    + (inner(qi, div(ui)) + k*qi*pr + ee*qi*pi)*dx
    #mui equation
    + jump(ui,n)*muiS*dS + inner(ui,n)*mui*ds
    )

x, y = SpatialCoordinate(mesh)

f = exp(-((x-0.5)**2 + (y-0.5)**2)/0.25**2)

L = qi*f*dx

U = Function(W)

HProblem = LinearVariationalProblem(a, L, U,
                                    constant_jacobian=False)

condensed_params_mumps = {'ksp_type':'preonly',
                          'pc_type':'lu',
                          'pc_factor_mat_solver_type':'mumps'}

condensed_params_gt = {'ksp_type': 'fgmres',
                       'ksp_max_it': 100,
                       'mat_type': 'matfree',
                       'ksp_monitor': None,
                       'ksp_converged_reason': None,
                       'ksp_rtol': 1e-8,
                       'ksp_atol': 1e-30,
                       'pc_type': 'python',
                       'pc_python_type': 'firedrake.GTMGPC',
                       'gt': {'mg_levels': {'ksp_type': 'gmres',
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type:': 'python',
                                            'sub_pc_python_type': 'firedrake.ASMStarPC',
                                            'sub_pc_star_contruct_codim': 0,
                                            'ksp_monitor': None,
                                            #'ksp_converged_reason': None,
                                            #'ksp_max_it': 3
                       },
                              'mg_coarse': {'ksp_type': 'preonly',
                                            'pc_type': 'bjacobi',
                                            'sub_pc_type:': 'lu',
                                            'sub_pc_factor_mat_solver_type':'mumps'}
                              }}}
                                       
hybrid_params = {'mat_type': 'matfree',
                 'pmat_type': 'matfree',
                 'ksp_type':'gmres',
                 'ksp_converged_reason':None,
                 'pc_type': 'python',
                 'pc_python_type':  'firedrake.SCPC',
                 'pc_sc_eliminate_fields': '0, 1',
                 'condensed_field':condensed_params_gt}

appctx = {'get_coarse_operator': p1_callback,
          'get_coarse_space': get_p1_space}

def get_p1_space():
    return VectorFunctionSpace(mesh, "CG", 1, vfamily="R", vdegree=0)

def p1_callback():
    P1 = get_p1_space()
    p = TrialFunction(P1)
    q = TestFunction(P1)
    return inner(grad(p), grad(q))*dx

HSolver = LinearVariationalSolver(HProblem, solver_parameters=hybrid_params)

file0 = File('hh.pvd')

u_out = Function(V)
p_out = Function(Q)

for k0 in range(20):
    k.assign(k0)
    U.assign(0.)
    HSolver.solve()
