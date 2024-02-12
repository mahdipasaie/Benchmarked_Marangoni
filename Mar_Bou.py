import fenics as fe
import dolfin as df
import numpy as np
from fenics import (
    UserExpression, Constant, FunctionSpace, TestFunctions, 
    Function, MixedElement, RectangleMesh, MeshFunction, 
    cells, refine, Measure, SubDomain, VectorElement, 
    FiniteElement, derivative, NonlinearVariationalProblem, 
    NonlinearVariationalSolver, Point, DirichletBC, split, 
    near, TrialFunction, LogLevel, set_log_level, sqrt, VectorFunctionSpace, interpolate
)
from tqdm import tqdm
from mpi4py import MPI

set_log_level(LogLevel.ERROR)


########################## Tracking Information Functions and Dimenssionless Numbers ##################

def compute_global_velocity_extremes(upT, W, comm):

    # Define the dofmap for velocity
    dm0 = W.sub(0).dofmap()

    # Compute local max and min velocities
    u_max_local = np.abs(upT.vector().vec()[dm0.dofs()]).max()
    u_min_local = np.abs(upT.vector().vec()[dm0.dofs()]).min()

    # Compute global max and min velocities
    u_max = comm.allreduce(u_max_local, op=MPI.MAX)
    u_min = comm.allreduce(u_min_local, op=MPI.MIN)

    return u_max, u_min

def calculate_dimensionless_numbers(u_max, domain_length_x, K1, RHO1, MU1):

    # Calculate Peclet number (Advective/Diffusive transport rate)
    peclet_number = (u_max * domain_length_x) / K1

    # Calculate Reynolds number
    reynolds_number = RHO1 * u_max * domain_length_x / MU1

    return peclet_number, reynolds_number


def write_to_file(line_points , velocity_values ): 

    gathered_line_points = comm.gather(line_points, root=0)
    gathered_velocity_values = comm.gather(velocity_values, root=0)

    if rank == 0:
        # Concatenate data from all processors
        all_line_points = np.concatenate(gathered_line_points)
        all_velocity_values = np.concatenate(gathered_velocity_values)

        # Now you can save this data or process it further
        np.save('all_line_points.npy', all_line_points)
        np.save('all_velocity_values.npy', all_velocity_values)


    return 




#############################  END  ################################

#################### Define Parallel Variables ####################


# Get the global communicator
comm = MPI.COMM_WORLD

# Get the rank of the process
rank = comm.Get_rank()

# Get the size of the communicator (total number of processes)
size = comm.Get_size()

#############################  END  ################################


##################### Physical Constants ################################
    
GRAVITY = -10  # Acceleration due to gravity (m/s^2)
RHO1 = 760  # Fluid density (kg/m^3)
MU1 = 4.94 * 10 ** -4  # Dynamic viscosity (Pa.s)
K1 = 0.1  # Thermal conductivity (W/m.K)
CP1 = 2090  # Heat capacity (J/kg.K)
ALPHA1 = 1.3 * 10**-3  # Thermal expansion coefficient (1/K)
GAMMA = -8 * 10 ** -5  # Surface tension temperature derivative (N/m.K)

# Temperature Constants:
T_REF = 273.15  # Reference temperature (K)
T_RIGHT = 273.15  # Temperature on the right boundary (K)
DELTA_T = 2  # Temperature difference (K)
T_LEFT = T_RIGHT + DELTA_T  # Temperature on the left boundary (K)

#############################  END  ################################

##################### Mesh Refinement Functions For Bounderies ######################

def refine_mesh_near_boundary(mesh, threshold, domain):

    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = domain
    
    # Initialize a MeshFunction for marking cells to refine
    marker = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    # Iterate through each cell in the mesh
    for idx, cell in enumerate(cells(mesh)):
        x_mid, y_mid = cell.midpoint().x(), cell.midpoint().y()

        # Calculate the distance from the cell's midpoint to the boundary
        dist_to_left_boundary = abs(x_mid - X0)
        dist_to_right_boundary = abs(x_mid - X1)
        dist_to_bottom_boundary = abs(y_mid - Y0)
        dist_to_top_boundary = abs(y_mid - Y1)

        # Mark cells for refinement if they're within the threshold distance from any boundary
        if (min(dist_to_left_boundary, dist_to_right_boundary) < threshold or
            min(dist_to_bottom_boundary, dist_to_top_boundary) < threshold):
            marker.array()[idx] = True

    # Refine the mesh based on the marked cells
    refined_mesh = refine(mesh, marker)

    return refined_mesh


def refine_mesh_near_corners(mesh, threshold, domain):

    
    # Unpack domain coordinates
    (X0, Y0), (X1, Y1) = domain
    
    # Initialize a MeshFunction for marking cells to refine
    marker = MeshFunction("bool", mesh, mesh.topology().dim(), False)

    # Iterate through each cell in the mesh
    for idx, cell in enumerate(cells(mesh)):
        x_mid, y_mid = cell.midpoint().x(), cell.midpoint().y()

        # Calculate the distance from the cell's midpoint to the corners
        dist_to_bottom_left_corner = sqrt((x_mid - X0)**2 + (y_mid - Y0)**2)
        dist_to_bottom_right_corner = sqrt((x_mid - X1)**2 + (y_mid - Y0)**2)
        dist_to_top_left_corner = sqrt((x_mid - X0)**2 + (y_mid - Y1)**2)
        dist_to_top_right_corner = sqrt((x_mid - X1)**2 + (y_mid - Y1)**2)

        # Mark cells for refinement if they're within the threshold distance from any corner
        if ( dist_to_top_left_corner < threshold):
            marker.array()[idx] = True

    # Refine the mesh based on the marked cells
    refined_mesh = refine(mesh, marker)

    return refined_mesh


#############################  END  ################################


############################## Define domain sizes and discretization parameters ################################



# Define grid spacing in x and y directions (meters)

grid_spacing_x = 0.1e-3  # 0.1 mm converted to meters
grid_spacing_y = 0.1e-3  # 0.1 mm converted to meters

# Define time step for the simulation (arbitrary units)
dt = 1000



# Adjust the domain length to ensure it is divisible by the grid spacing and slightly larger than the desired size

domain_length_x = 10e-3
domain_length_y = 5e-3


# Calculate the number of divisions along each axis based on approximate domain size and grid spacing
num_divisions_x = int( domain_length_x  / grid_spacing_x)
num_divisions_y = int( domain_length_y / grid_spacing_y)


# Define the origin point of the domain (bottom left corner)
origin = df.Point(0.0, 0.0)

# Calculate the top right corner based on the origin and adjusted domain lengths
top_right_corner = df.Point(origin.x() + domain_length_x, origin.y() + domain_length_y)

# Create the initial rectangular mesh using the defined corners and number of divisions
initial_mesh = fe.RectangleMesh(origin, top_right_corner, num_divisions_x, num_divisions_y)

# Define Domain 

Domain = [ ( 0.0 , 0.0 ) ,( 0.0 + domain_length_x , 0.0 + domain_length_y ) ]

#############################  END  ################################

############################ Modify Initial Mesh ######################

mesh = initial_mesh

mesh  = refine_mesh_near_boundary( mesh, 0.2e-3, Domain )
mesh  = refine_mesh_near_boundary( mesh, 0.2e-3, Domain )


mesh  = refine_mesh_near_corners( mesh, 0.3e-3, Domain  )


#############################  END  ################################


######################################################################

def create_function_spaces(mesh):


    # Define finite elements for velocity, pressure, and temperature
    P2 = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)  # Velocity
    P1 = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)  # Pressure 
    PT = fe.FiniteElement( "Lagrange", mesh.ufl_cell(), 1  )#temperature

    # Define mixed elements
    element = MixedElement([P2, P1, PT])

    # Create a function space
    W = FunctionSpace(mesh, element)

    # Define test functions
    v_test, q_test, s_test = TestFunctions(W)

    # Define current and previous solutions
    upT = Function(W)  # Current solution
    upT0 = Function(W)  # Previous solution

    # Split functions to access individual components
    u_answer, p_answer, T_answer = split(upT)  # Current solution
    u_prev, p_prev, T_prev = split(upT0)  # Previous solution

    return W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev

# Usage example:
# W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev = create_function_spaces(mesh)

#############################  END  ################################


############################ Defining Equations ###########################

# Related Functions for defining equaions
def epsilon(u):  

    return 0.5 * (fe.grad(u) + fe.grad(u).T)

def sigma(u, p, mu1):

    return 2 * mu1 * epsilon(u) - p * fe.Identity(len(u))

def Traction(T, n_v, gamma):

    return gamma * (fe.grad(T) - fe.dot(n_v, fe.grad(T)) * n_v)


# main equaions

def F1(u_answer, q_test, dt):

    F1 = fe.inner(fe.div(u_answer), q_test) * dt * fe.dx

    return F1

def F2(u_answer, u_prev, p_answer, T_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1):

    global GRAVITY, T_REF
    
    F2 = (
        fe.inner((u_answer - u_prev) / dt, v_test) * fe.dx
        + fe.inner(fe.dot(u_answer, fe.grad(u_answer)), v_test) * fe.dx
        + (1/rho1) * fe.inner(sigma(u_answer, p_answer, mu1), epsilon(v_test)) * fe.dx
        - (1/rho1) * fe.inner(Traction(T_answer, n_v, gamma), v_test) * ds1(1)
        # Uncomment the following lines if buoyancy force is needed
        + fe.inner(GRAVITY * alpha1 * (T_answer - T_REF), v_test[1]) * fe.dx  # Bouyancy y-component

        # Remeber alpha1 ?!
    )

    return F2

def F3(T_answer, T_prev, u_answer, s_test, dt, rho1, Cp1, K1):


    F3 = ( fe.inner((T_answer - T_prev) / dt, s_test) * fe.dx
          + fe.inner(fe.grad(s_test), K1/(rho1 * Cp1) * fe.grad(T_answer)) * fe.dx
          +   fe.inner(s_test, fe.dot(u_answer, fe.grad(T_answer))) * fe.dx)

    return F3


def solve_navier_stokes_heat_transfer(mesh, Bc, dt, upT, W, rho1, mu1, gamma, n_v, alpha1, Cp1, K1, absolute_tolerance, relative_tolerance, u_answer, u_prev, T_answer, T_prev, p_answer, v_test, q_test, s_test, ds1, dx1):
 

    # Define weak forms
    F1_form = F1(u_answer, q_test, dt)
    F2_form = F2(u_answer, u_prev, p_answer, T_answer, v_test, dt, rho1, n_v, mu1, gamma, alpha1, ds1, dx1)
    F3_form = F3(T_answer, T_prev, u_answer, s_test, dt, rho1, Cp1, K1)

    # Define the combined weak form
    L = F1_form + F2_form + F3_form

    # Define the Jacobian
    J = derivative(L, upT)

    # Set up the nonlinear variational problem
    problem = NonlinearVariationalProblem(L, upT, Bc, J)

    # Set up the solver
    solver = NonlinearVariationalSolver(problem)

    # Set solver parameters
    prm = solver.parameters
    prm['newton_solver']['relative_tolerance'] = relative_tolerance
    prm['newton_solver']['absolute_tolerance'] = absolute_tolerance
    prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True



    return solver


#############################  END  ########################################

############################ Boundary Condition Section #################

def Define_Boundary_Condition(W, Domain, T_LEFT, T_RIGHT  ) : 
    # Define the Domain boundaries based on the previous setup
    (X0, Y0), (X1, Y1) = Domain

    # Define boundary conditions for velocity, pressure, and temperature
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X0)

    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], X1)

    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y0)

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], Y1)

    # Instantiate boundary classes
    left_boundary = LeftBoundary()
    right_boundary = RightBoundary()
    bottom_boundary = BottomBoundary()
    top_boundary = TopBoundary()

    # Define Dirichlet boundary conditions
    bc_u_left = DirichletBC(W.sub(0), Constant((0, 0)), left_boundary)
    bc_u_right = DirichletBC(W.sub(0), Constant((0, 0)), right_boundary)
    bc_u_bottom = DirichletBC(W.sub(0), Constant((0, 0)), bottom_boundary)
    bc_u_top = DirichletBC(W.sub(0).sub(1), Constant(0), top_boundary)

    bc_T_left = DirichletBC(W.sub(2), T_LEFT, left_boundary)
    bc_T_right = DirichletBC(W.sub(2), T_RIGHT, right_boundary)

    # Point for setting pressure
    zero_pressure_point = Point( X0  ,  Y1 )
    bc_p_zero = DirichletBC(W.sub(1), Constant(0), lambda x, on_boundary: near(x[0], zero_pressure_point.x()) and near(x[1], zero_pressure_point.y()), method="pointwise")


    # Combine all boundary conditions

    bc_all = [bc_u_left, bc_u_right, bc_u_bottom, bc_u_top, bc_T_left, bc_T_right, bc_p_zero]

    # ******************************************
    # Create a MeshFunction for marking the subdomains


    sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    sub_domains.set_all(0)

    # Mark the subdomains with the boundary objects
    bottom_boundary.mark(sub_domains, 2)  # Mark the bottom boundary with label 2
    top_boundary.mark(sub_domains, 1)  # Mark the top boundary with label 1

    # Define measures with the subdomain marking
    ds = Measure("ds", domain=mesh, subdomain_data=sub_domains)  # For boundary integration

    # Define an interior domain class to mark the interior of the domain
    class Interior(SubDomain):
        def inside(self, x, on_boundary):
            return not (top_boundary.inside(x, on_boundary) or bottom_boundary.inside(x, on_boundary))

    # Mark the interior domain
    domains2 = MeshFunction("size_t", mesh, mesh.topology().dim())
    domains2.set_all(0)  # Initially mark all cells as 0
    interior_obj = Interior()
    interior_obj.mark(domains2, 1)  # Mark cells inside the interior domain as 1

    # Define the dx measure for the interior domain
    dx = Measure("dx", domain=mesh, subdomain_data=domains2)

    return ds, dx, bc_all

#############################  END  ################################





#################### Define Step 1 For Solving  ####################

W, v_test, q_test, s_test, upT, upT0, u_answer, p_answer, T_answer, u_prev, p_prev, T_prev = create_function_spaces(mesh)

n_v = Constant( ( 0, 1 ) )

ds1, dx1, bc_all = Define_Boundary_Condition(W, Domain, T_LEFT, T_RIGHT  )



# solver = solve_navier_stokes_heat_transfer(
#     mesh, bc_all, dt, upT, W, RHO1, MU1, GAMMA, n_v, ALPHA1, CP1, K1, 1E-5, 1E-6 )

solver = solve_navier_stokes_heat_transfer(
    mesh, bc_all, dt, upT, W, RHO1, MU1, GAMMA, n_v, ALPHA1, CP1, K1,  1E-8 , 1E-7,
      u_answer, u_prev, T_answer, T_prev, p_answer, v_test, q_test, s_test, ds1, dx1)



#############################  END  ###############################



#################### Define Initial Condition ####################

class InitialConditions(fe.UserExpression):

    def eval(self, values, x):

        values[0] = 0  # Initial x-component of velocity
        values[1] = 0  # Initial y-component of velocity
        values[2] = 0.0  # Initial pressure
        values[3] = 273.15  # Initial temperature (in Kelvin)

    def value_shape(self):
 
        return (4,)

initial_v  = InitialConditions( degree = 2 ) 

upT.interpolate( initial_v )
upT0.interpolate( initial_v )

#############################  END  ################################

############################ File Section #########################


file = fe.XDMFFile("Mar_Bou.xdmf" ) # File Name To Save #


def write_simulation_data(Sol_Func, time, file, variable_names ):


    
    # Configure file parameters
    file.parameters["rewrite_function_mesh"] = True
    file.parameters["flush_output"] = True
    file.parameters["functions_share_mesh"] = True

    # Split the combined function into its components
    functions = Sol_Func.split(deepcopy=True)

    # Check if the number of variable names matches the number of functions
    if variable_names and len(variable_names) != len(functions):
        raise ValueError("The number of variable names must match the number of functions.")

    # Rename and write each function to the file
    for i, func in enumerate(functions):
        name = variable_names[i] if variable_names else f"Variable_{i}"
        func.rename(name, "solution")
        file.write(func, time)

    file.close()



T = 0

variable_names = [  "Vel", "Press", "T" ]  # Adjust as needed


write_simulation_data( upT0, T, file , variable_names=variable_names )


#############################  END  ###############################


########################### Solving Loop  #########################



# Time-stepping loop
for it in tqdm(range(200000)):





    # Write data to file at certain intervals
    if it % 100 == 0:
        write_simulation_data(upT, T, file, variable_names)


    # Solve the system
    no_of_it, converged = solver.solve()

    # Update the previous solution
    upT0.vector()[:] = upT.vector()


    # Update time
    T = T + dt


    # Printing Informations Related to solutions behaviour

    u_max, u_min = compute_global_velocity_extremes(upT, W, comm)
    peclet_number, reynolds_number = calculate_dimensionless_numbers(u_max, domain_length_x, K1, RHO1, MU1)
    
    if rank == 0 and it% 1000 ==0  :  # Only print for the root process

        print(" ├─ Iteration: " + str(it), flush=True)
        print(" Peclet Number is (Advective/Diffusive) Transport rate: " + str(peclet_number) , flush=True)
        print(" Reynolds Number is: " + str(reynolds_number), flush=True)
        print(" CFL Condition : " +   str( ( u_max * dt / domain_length_x )  > 1  ) , flush=True )

#############################  END  ###############################
