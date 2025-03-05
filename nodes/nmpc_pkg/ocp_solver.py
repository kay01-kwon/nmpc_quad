import os
import shutil
import sys
from acados_template import AcadosOcp, AcadosOcpSolver
from quad_model import QuadModel
import scipy.linalg
import numpy as np

# Initial state
X0 = np.array([
    0.0, 0.0, 0.0,          # position
    0.0, 0.0, 0.0,          # velocity
    1, 0.0, 0.0, 0.0,     # quaternion
    0.0, 0.0, 0.0           # angular velocity
])


class OcpSolver():
    def __init__(self, u_min = 0., u_max = 5, n_nodes = 10, t_horizon = 1.0):
        '''
        Constructor for OcpSolver
        :param u_min: minimum rotor thrust (N)
        :param u_max: maximum rotor thrust (N)
        :param n_nodes: Number of nodes for NMPC
        :param T_horizon: Prediction horizon
        '''

        # If there exist c_generated_code and acados json file
        # remove it.
        nmpc_pkg_dir = os.getcwd()
        print(nmpc_pkg_dir)
        node_dir = os.path.dirname(nmpc_pkg_dir)
        nmpc_quad_dir = os.path.dirname(node_dir)
        prev_dir = os.path.dirname(nmpc_quad_dir)

        acados_json_dir = os.path.join(prev_dir, "acados_ocp_nlp.json")
        acados_c_dir = prev_dir + '/c_generated_code'

        print('Acados json directory: ',acados_json_dir)
        print('Acados c generated directory: ',acados_c_dir)

        if os.path.exists(acados_json_dir):
            os.remove(acados_json_dir)

        if os.path.exists(acados_c_dir):
            shutil.rmtree(acados_c_dir)

        acados_home_json_dir = os.path.join('/home/kay','acados_ocp_nlp.json')
        acados_c_dir = '/home/kay/c_generated_code'

        if os.path.exists(acados_home_json_dir):
            os.remove(acados_home_json_dir)

        if os.path.exists(acados_c_dir):
            shutil.rmtree(acados_c_dir)


        # Create AcadosOcp
        self.ocp = AcadosOcp()

        # Object generation
        quad_model_obj = QuadModel(m = 0.716,
                                J =np.array([0.007, 0.007, 0.012]),
                                l = 0.17,
                                C_moment = 0.05,
                                model_description = '+')

        # Get Quad model from the quad_model_obj
        self.model = quad_model_obj.get_acados_model()

        # Set ocp model
        self.ocp.model = self.model

        # Set min max value of the rotor speed
        self.u_min = u_min
        self.u_max = u_max

        self.u_prev = np.zeros((4,))

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
        # cost Q
        # px py pz
        # vx vy vz
        # qw qx qy qz
        # wx wy wz
        self.Q_mat = np.diag([1, 1, 1,
                              0.5, 0.5, 0.5,
                              0, 0.5, 0.5, 0.5,
                              0.05, 0.05, 0.05])

        # cost R:
        # u1, u2, u3, u4 (RPM)
        self.R_mat = 0.01*np.diag([1.0, 1.0, 1.0, 1.0])

        # Set cost type for OCP
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # Cost setup for state
        self.ocp.cost.Vx = np.zeros((self.ny, self.nx))
        self.ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)
        self.ocp.cost.Vx_e = np.eye(self.nx)

        # Cost setup for control input
        self.ocp.cost.Vu = np.zeros((self.ny, self.nu))
        self.ocp.cost.Vu[-4:, -4:] = np.eye(self.nu)

        # Weight setup
        self.ocp.cost.W = scipy.linalg.block_diag(self.Q_mat, self.R_mat)
        self.ocp.cost.W_e = self.Q_mat

        # Reference setup
        self.ocp.cost.yref = np.concatenate((X0, np.zeros(4)))
        self.ocp.cost.yref_e = X0

    def set_ocp_constraints(self):
        '''
        Set constraints
        :return: None
        '''
        # Constraint on initial state
        self.ocp.constraints.x0 = X0

        # Constraint on control input
        self.ocp.constraints.lbu = np.array([self.u_min] * 4)
        self.ocp.constraints.ubu = np.array([self.u_max] * 4)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    def set_ocp_solver_options(self):
        '''
        Set solver options
        :return: None
        '''
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.print_level = 0     # Do not print out
        self.ocp.solver_options.nlp_solver = "SQP_RTI"

        self.ocp.solver_options.tf = self.T_horizon

    def ocp_solve(self, state, ref):
        '''
        Set ocp solver (State and reference)
        :param state: Initial state of p_xyz, v_xyz, q_wxyz, w_xyz
        :param ref: p_xyz_ref, v_xyz_ref, q_wxyz, w_xyz
        :return: u
        '''

        y_ref = np.concatenate((ref, np.zeros((self.nu,))))
        y_ref_N = ref

        # Fill initial state
        self.acados_ocp_solver.set(0,"lbx", state)
        self.acados_ocp_solver.set(0, "ubx", state)

        for stage in range(self.ocp.dims.N):
            self.acados_ocp_solver.set(stage, "y_ref", y_ref)
        self.acados_ocp_solver.set(self.ocp.dims.N,"y_ref", y_ref_N)

        status = self.acados_ocp_solver.solve()

        u = self.acados_ocp_solver.get(0,"u")

        self.u_prev = u

        return status, u