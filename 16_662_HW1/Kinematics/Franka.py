import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const) <-> relative between joints
        self.Tjoint = []  # Transforms for each joint (init eye) <-> defined between joints (wrt previous joint frame)
        self.Tcurr = []  # Coordinate frame of current (init eye) <-> defined between base and current joint (global base frame)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])
            self.Tjoint.append([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]])

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.] # joint angles

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''
        self.q[0:-1] = ang

        for i in range(8):
            R = rt.MatrixExp(self.axis[i], self.q[i])
            self.Tjoint[i] = R
            if i == 0:
                self.Tcurr[i] = np.matmul(self.Tlink[i], self.Tjoint[i])
            else:
                self.Tcurr[i] = np.matmul(self.Tcurr[i-1], np.matmul(self.Tlink[i], self.Tjoint[i]))
        
        last=self.Tcurr[-1][:3,-1]
        for i in range(len(self.Tcurr) - 1):
            T=self.Tcurr[i]
            a=T[0:3,2]
            p=last-T[0:3,-1]
            self.J[0:3,i]=np.cross(a,p)
            self.J[3:,i]=a
        # print("Jacobian\n",np.round(self.J,2))
        return self.Tcurr, self.J


    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''
        Error_ret = 0
        self.q[0:-1] = ang 
        max_iterations = 1000
        
        C = np.array([
            [1000000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1000000.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1000000.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1000.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1000.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1000.0]
        ])

        W = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
        ])

        # self.Tcurr,self.J = self.ForwardKin(ang)
        for _ in range(max_iterations):
                    
            # run FK here
            Tcurr, J = self.ForwardKin(ang)

            Rerr = TGoal[:3, :3]@np.array(Tcurr[-1])[:3, :3].T 
            axis, angle = rt.R2axisang(Rerr)
            angle = np.clip(angle, -10*r_eps, 10*r_eps)

            Verr = np.array(axis)*angle

            Terr = (TGoal[:3, 3] - np.array(Tcurr[-1])[:3, 3]) 
            Terr=np.clip(Terr, -10*x_eps, 10*x_eps)            
            Err = np.concatenate((Terr, Verr))
            if np.linalg.norm(Err[:3]) < x_eps and np.linalg.norm(Err[3:]) < r_eps:
                break
            
            J_hash = np.linalg.inv(W) @ J.T @ np.linalg.inv(J @ np.linalg.inv(W) @ J.T + np.linalg.inv(C))
            dq = J_hash @ Err
            ang += dq
            
        Error_ret = Err

        return ang, Error_ret

