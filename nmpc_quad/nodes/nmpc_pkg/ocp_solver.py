from acados_template import AcadosOcp, AcadosOcpSolver
import quad_model
import scipy.linalg
import numpy as np
import casadi as cs

# Initial state
X0 = np.array([
    0.0, 0.0, 0.0,          # position
    0.0, 0.0, 0.0,          # velocity
    0.0, 0.0, 0.0, 1.0,     # quaternion
    0.0, 0.0, 0.0           # angular velocity
])


class OcpSolver():
    def __init__(self, u_min = 100, u_max = 1000 ,n_nodes = 20, t_horizon = 2.0):
        '''
        Constructor for OcpSolver
        :param u_min: minimum rotor speed
        :param u_max: maximum rotor speed
        :param n_nodes: Number of nodes for NMPC
        :param T_horizon: Prediction horizon
        '''

        # Create AcadosOcp
        self.ocp = AcadosOcp()

        # Object generation
        quad_model_obj = quad_model.QuadModel(m = 1,
                                          J =np.diag([1.0, 1.0, 1.0]),
                                          l = 0.2,
                                          C_lift = 1,
                                          C_moment = 1,
                                          model_description = '+')

        # Get Quad model from the quad_model_obj
        self.model = quad_model_obj.get_acados_model()

        # Set ocp model
        self.ocp.model = self.model

        # Set min max value of the rotor speed
        self.u_min = u_min
        self.u_max = u_max

        # Set horizon
        self.N = n_nodes
        self.ocp.dims.N = self.N

        self.T_horizon = t_horizon

        # Get the dimension of state, input, y and y_e (terminal)
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.ny = self.nx + self.nu
        self.ny_e = self.nx

        # Set ocp cost
        self.set_ocp_cost()

        # Set constraint for state and input
        self.set_ocp_constraints()

        # Set solver options
        self.set_ocp_solver_options()

        # Create ocp solver
        self.acados_ocp_solver = AcadosOcpSolver(self.ocp)



    def set_ocp_cost(self):
        '''
        Set OCP cost
        :return:
        '''
        # cost Q:
        # px, py, pz,
        # vx, vy, vz,
        # qx, qy, qz, qw(Ignore)
        # wx, wy, wz
        self.Q_mat = np.diag([1.0, 1.0, 1.0,
                              0.05, 0.05, 0.05,
                              0.1, 0.1, 0.1, 0,
                              0.05, 0.05, 0.05])

        # cost R:
        # u1, u2, u3, u4 (RPM)
        self.R_mat = np.diag([0.1, 0.1, 0.1, 0.1])

        # Set cost type for OCP
        self.ocp.cost.cost_type = 'Linear_LS'
        self.ocp.cost.cost_type_e = 'Linear_LS'

        # Cost setup for state
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

        # Cost setup for control input
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[-4:, -4:] = np.eye(self.nu)

        # Weight setup
        self.ocp.cost.W = scipy.linalg.block_diag(self.Q_mat, self.R_mat)
        self.ocp.cost.W_e = self.Q_mat

        # Reference setup
        self.ocp.cost.yref = np.concatenate((X0, np.zeros(4)))
        self.ocp.cost.yref_e = np.zeros((self.ny,))

    def set_ocp_constraints(self):
        '''
        Set constraints
        :return: None
        '''
        # Constraint on initial state
        self.ocp.constraints.x0 = X0

        # Constraint on control input
        self.ocp.constraints.lbu = np.array([self.u_min] ** 4)
        self.ocp.constraints.ubu = np.array([self.u_max] ** 4)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    def set_ocp_solver_options(self):
        '''
        Set solver options
        :return: None
        '''
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_solver = "GAUSSIAN_NEWTON"
        self.ocp.solver_options.integrand_solver = "ERK"
        self.ocp.solver_options.nlp_solver = "SQP_RTI"

        self.ocp.solver_options.tf = self.T_horizon

    def set_ocp_solver(self, state, ref):
        '''
        Set ocp solver (State and reference)
        :param state: Initial state of p_xyz, q_xyzw, v_xyz, w_xyz
        :param y_ref: p_xyz_ref, q_xyzw_ref, v_xyz_ref, w_xyz_ref, u_ref
        :return: u
        '''

        y_ref = np.concatenate(state, np.zeros(self.nu))
        y_ref_N = ref

        # Fill in initial state
        self.acados_ocp_solver.set(0 ,"lbx",state)
        self.acados_ocp_solver.set(0, "ubx",state)

        for i in range(self.ocp.dims.N):
            self.acados_ocp_solver.set(i, "y_ref", y_ref)
        self.acados_ocp_solver.set(self.ocp.dims.N,"y_ref", y_ref_N)

        status = self.acados_ocp_solver.solve()

        u = self.acados_ocp_solver.get(0,"u")

        return u






