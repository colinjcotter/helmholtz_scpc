from firedrake import *
#A real-mode Helmholtz discretisation
#using augmented Lagrangian formulation

n = 50
mesh = PeriodicUnitSquareMesh(n,n)

k = Constant(5.)
ee = Constant(0.0)
gamma = Constant(100.0)

deg = 1
V = VectorFunctionSpace(mesh, "BDM", deg)
Q = VectorFunctionSpace(mesh, "DG", deg-1)
W = V*Q

u, p = TrialFunctions(W)
ur = u[0,:]
ui = u[1,:]
pr = p[0]
pi = p[1]

v, q = TestFunctions(W)
vr = v[0,:]
vi = v[1,:]
qr = q[0]
qi = q[1]

# -i*((ee+ik)*<v,u> - <div(v), p> +
#      ik*gamma*((ee+ik)*<div(v),p> + <div(u),div(v)>))
# + (ee+ik)*<q,p> + <q,div(u)>

a = (
    #ur equation
    (ee*inner(vr,ur) - k*inner(vr, ui) - inner(div(vr), pr))*dx
    #pr equation (and associated contribution to ui equation)
    + (inner(qr, div(ur)) - k*qr*pi + ee*qr*pr)*dx
    + gamma*k*(inner(div(vi), div(ur)) - k*div(vi)*pi + ee*div(vi)*pr)*dx
    #ui equation
    + (ee*inner(vi,ui) + k*inner(vi, ur) - inner(div(vi), pi))*dx
    #pi equation (and associated contribution to ur equation)
    + (inner(qi, div(ui)) + k*qi*pr + ee*qi*pi)*dx
    - gamma*k*(inner(div(vr), div(ui)) + k*div(vr)*pr + ee*div(vr)*pi)*dx
    )

x, y = SpatialCoordinate(mesh)

f = exp(-((x-0.5)**2 + (y-0.5)**2)/0.1**2)

L = qi*f/k*dx

U = Function(W)

HProblem = LinearVariationalProblem(a, L, U)

sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    'ksp_monitor': None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright = {
    "ksp_type": "gmres",
    "ksp_max_it": 10,
    "ksp_monitor": None,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

sparameters["fieldsplit_1"] = bottomright

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

topleft_MG = {
    "ksp_type": "preonly",
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

ctx = {"mu": -1.0/gamma/k}

sparameters["fieldsplit_0"] = topleft_LU

HSolver = LinearVariationalSolver(HProblem, solver_parameters=sparameters,
                                  appctx=ctx)

file0 = File('hh.pvd')

HSolver.solve()
u, p = U.split()
ur = u.sub(0)
ui = u.sub(1)
pr = p.sub(0)
pi = p.sub(1)
file0.write(ur, ui, pr, pi)
