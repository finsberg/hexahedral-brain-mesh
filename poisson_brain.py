from mpi4py import MPI
import numpy as np
import ufl
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import pyvista

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "brain.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Mesh")
    # ct = xdmf.read_meshtags(mesh, name="Mesh", attribute_name="Cell markers")


V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = dolfinx.fem.dirichletbc(np.array((0.0), dtype=np.float64), boundary_dofs, V=V)

f = dolfinx.fem.Constant(mesh, 1.0)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx


print("Assembling system...")
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Poisson",
)
print("Solving system...")
uh = problem.solve()

print("Saving solution to file...")
with dolfinx.io.VTXWriter(mesh.comm, "poisson.bp", [uh], engine="BP4") as vtx:
    vtx.write(0.0)


u_topology, u_cell_types, u_geometry = dolfinx.plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter(off_screen=True)
u_plotter.add_mesh_clip_plane(u_grid, show_edges=True, crinkle=True, normal="-z")
u_plotter.view_xy()
u_plotter.screenshot("poisson_brain.png")
