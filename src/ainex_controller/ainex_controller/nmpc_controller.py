import casadi as ca
import numpy as np

class NMPC:
    def __init__(
            self, 
            T_HORIZON_s = 1.0,
            N = 20,
            
            n_joints = 4,
            n_task_coords = 2,
            
            Q_diag = [100.0, 100.0, 10.0],
            R_diag = [0.1, 0.1, 0.1, 0.1],
            
            theta_dot_max = 2.0,  
            theta_min = np.array([-np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2]),
            theta_max = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2]),
            
            homogeneous_transform_params = {
                'T_0_1':     ([0,0,0], '-y'),
                'T_1_2':     ([0.02000, -0.02151, 0], 'x'),
                'T_2_3':     ([-0.02000, -0.07411, 0], '-y'),
                'T_3_4':     ([0.00040, -0.01702, 0.01907], 'z'),
                'T_4_wrist': ([1,1,1], 'non'),
            },

            solver_options = {
                'qpsol': 'qpoases', # Fast QP solver for MPC
                'qpsol_options': {'printLevel': 'none'},
                'print_time': False,
                'print_header': False,
                'print_status': False
            }

        ) -> None:

        """ Setup Variables """
        self.T_HORIZON_s = T_HORIZON_s      # Prediction horizon time (s)
        self.N = N                          # Number of control intervals
        self.dt = T_HORIZON_s / N           # Time step (delta t)

        self.n_joints = n_joints            # 0: Shoulder Pitch, 1: Shoulder Roll, 2: Elbow Pitch, 3: Elbow Yaw
        self.n_task_coords = n_task_coords  # 0: x, 1: y

        # Weight Matrices for tuning optimization
        self.Q_diag = Q_diag                #[x, y, theta_elbow]
        self.R_diag = R_diag                #[sho_p, sho_r, el_p, el_y]

        # Optimization Constraints
        self.theta_dot_max = theta_dot_max  # rad/s
        self.theta_min = theta_min          # Maximum negative joint rotation allowed
        self.theta_max = theta_max          # Maximum positive joint rotation allowed

        self.T_0_1_params     = homogeneous_transform_params['T_0_1']
        self.T_1_2_params     = homogeneous_transform_params['T_1_2']
        self.T_2_3_params     = homogeneous_transform_params['T_2_3']
        self.T_3_4_params     = homogeneous_transform_params['T_3_4']
        self.T_4_wrist_params = homogeneous_transform_params['T_4_wrist']

        self.define_forward_kinemtics()
        self.nmpc_formulation()

        self.opti.solver('sqpmethod', solver_options)

        self.prev_Theta = None
        self.prev_U = None

    def define_forward_kinemtics(self):
        # We define the chain relative to the shoulder.
        # Shoulder Pitch -> Shoulder Roll -> Elbow Pitch -> Elbow Yaw -> Wrist
        # The return function is a symbolic vector function, finding the origin of wrist relative to the shoulder
        # Input [theta] -> Output [x_wrist, y_wrist]

        def get_homogeneous_transform(xyz, axis, angle):
            
            # Translation between joints
            T = ca.SX.eye(4)
            T[0:3, 3] = xyz  
           
            # Rotation between joints 
            c = ca.cos(angle); s = ca.sin(angle)
            match axis:
                case 'x':  T[1,1]=c; T[1,2]=-s; T[2,1]= s; T[2,2]=c
                case '-x': T[1,1]=c; T[1,2]= s; T[2,1]=-s; T[2,2]=c

                case 'y':  T[0,0]=c; T[0,2]= s; T[2,0]=-s; T[2,2]=c
                case '-y': T[0,0]=c; T[0,2]=-s; T[2,0]= s; T[2,2]=c

                case 'z':  T[0,0]=c; T[0,1]=-s; T[1,0]= s; T[1,1]=c
                case '-z': T[0,0]=c; T[0,1]= s; T[1,0]=-s; T[1,1]=c

                case 'non': pass

            return T

        theta = ca.SX.sym('theta', self.n_joints)
        # 1. Shoulder Pitch, base of the chain
        T_0_1 = get_homogeneous_transform(self.T_0_1_params[0], self.T_0_1_params[1], theta[0])

        # 2. Shoulder Pitch -> Shoulder Roll 
        T_1_2 = get_homogeneous_transform(self.T_1_2_params[0], self.T_1_2_params[1], theta[1])

        # 3. Shoulder Roll -> Elbow Pitch
        T_2_3 = get_homogeneous_transform(self.T_2_3_params[0], self.T_2_3_params[1], theta[2])

        # 4. Elbow Pitch -> Elbow Yaw
        T_3_4 = get_homogeneous_transform(self.T_3_4_params[0], self.T_3_4_params[1], theta[3])

        # 5. Elbow Yaw -> Wrist (Only translations)
        T_4_wrist = get_homogeneous_transform(self.T_4_wrist_params[0], self.T_4_wrist_params[1], 0)    

        # Total Chain: Shoulder -> Wrist
        T_shoulder_wrist = ca.mtimes([T_0_1, T_1_2, T_2_3, T_3_4, T_4_wrist])

        # Extract Position (x, y)
        p_wrist = T_shoulder_wrist[0:2, 3]

        self.forward_kinematics_wrist = ca.Function('forward_kinematics_wrist', [theta], [p_wrist])

    def nmpc_formulation(self):
        self.opti = ca.Opti() # Casadi's Non-linear problem solver

        """ Decision Variables """
        self.Theta = self.opti.variable(self.n_joints, self.N+1) # Joint positions (Theta) for t = 0 to N
        self.U  = self.opti.variable(self.n_joints, self.N)      # Joint Velocities (u = Theta_dot) for t = 0 to N-1

        """ Parameters """
        self.Theta_0 = self.opti.parameter(self.n_joints)        # Initial Joint Configuration
        self.s_ref = self.opti.parameter(3)                      # Reference Trajectory for States s = [x, y, theta_elbow]

        """ Objective Function """
        cost = 0
        self.Q = np.diag(self.Q_diag)
        self.R = np.diag(self.R_diag)

        for k in range(self.N):
            """ Calculate current State """                 
            s_k  = ca.vertcat(                                   # s_k = [x_wrist, y_wrist, theta_elbow]
                self.forward_kinematics_wrist(self.Theta[:, k]), 
                self.Theta[2, k]
                )                                
            
            e_k = s_k - self.s_ref                               # Error
            
            # Accumulated Cost -> 0.5*e^T*Q*e + 0.5*u^T*R*u
            cost += 0.5 * ca.mtimes([e_k.T, self.Q, e_k]) + 0.5 * ca.mtimes([self.U[:,k].T, self.R, self.U[:,k]])  

        s_N  = ca.vertcat(
            self.forward_kinematics_wrist(self.Theta[:, self.N]), 
            self.Theta[2, self.N]
            )
        e_N  = s_N - self.s_ref
        cost += 0.5 * ca.mtimes([e_N.T, self.Q, e_N])            # Terminal cost

        self.opti.minimize(cost)

        """ Constraints """
        self.opti.subject_to(self.Theta[:, 0] == self.Theta_0)   # Initial Condition

        for k in range(self.N):
            self.opti.subject_to(self.Theta[:, k+1] == self.Theta[:, k] + self.dt * self.U[:, k])  # Kinematics using euler integration (theta_{k+1} = theta_k + dt * theta_dot_k)
            self.opti.subject_to(self.opti.bounded(-self.theta_dot_max, self.U[:, k], self.theta_dot_max)) # Velocity Limits
            
        self.opti.subject_to(self.opti.bounded(self.theta_min, self.Theta, self.theta_max))        # Joint Position Limits

    def solve_nmpc(self, current_Theta, current_s_ref):
        """ Set Parameters """
        self.opti.set_value(self.Theta_0, current_Theta)
        self.opti.set_value(self.s_ref, current_s_ref)
        
        if self.prev_Theta is None or self.prev_U is None:                              # Cold start first iteration
            self.opti.set_initial(self.Theta, np.tile(current_Theta, (self.N+1, 1)).T)  # Fill the whole horizon with the current state
            self.opti.set_initial(self.U, 0.0)

        else:                                                                           # Warm start, uses the previous solution as an initial guess to converge faster
            theta_guess = np.hstack((                                                   
                self.prev_Theta[:, 1:],                                                 # Drop index 0, and duplicate index N at the end
                self.prev_Theta[:, -1:]      
            ))
            
            u_guess = np.hstack((
                self.prev_U[:, 1:],                                                     # Drop index 0, and duplicate index N at the end
                self.prev_U[:, -1:]           
            ))
            
            self.opti.set_initial(self.Theta, theta_guess)
            self.opti.set_initial(self.U, u_guess)
        
        try:
            sol = self.opti.solve()

            # Save solution to use as warm start for the next iteration
            self.prev_Theta = sol.value(self.Theta)
            self.prev_U     = sol.value(self.U)
            
            u_opt = sol.value(self.U[:, 0])

            return u_opt
                    
        except Exception as e:
            print(f"Failed to solve NMPC problem: {e}")