import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion

# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_sine_board.xml"

################################# Control Callback Definitions #############################

def get_board_control(model, data):
    # Instantite a handle to the desired body on the robot
    omega = 0.4
    data.ctrl[7] = 0.15*np.sin(omega*data.time) + 0.15

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints

    data.ctrl[:7] = data.qfrc_bias[:7]
    # get_board_control(model, data)

# Force control callback
def force_control(model, data):
    # Copy the code that you used to implement the controller from Part1 here
    # Instantite a handle to the desired body on the robot
    body=data.body("hand")

    # Get the Jacobian for the desired location on the robot (The end-effector)
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mj.mj_jacBody(model, data, Jp, Jr, body.id)
    J = np.vstack((Jp, Jr))  # 6 x nv Jacobian
    
    # Specify the desired force in global coordinates
    # F=np.array([15.8,0,0,0,0,0])
    F=np.array([15,0,0,0,0,0])

    # Compute the required control input using desied force values
    tau=J.T @ F

    # Set the control inputs
    data.ctrl[:7] = data.qfrc_bias[:7] + tau[:7]
    # print(body.xpos)
    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

    # Update control inputs to the whiteboard
    get_board_control(model, data)

# Control callback for an impedance controller
def impedance_control(model, data):
    # Copy the code that you used to implement the controller from Part1 here

    # Instantite a handle to the desired body on the robot
    body=data.body("hand")
    F_desired = np.array([15.0, 0.0, 0.0]) 
    K_p = F_desired / 5
    K_d = 1
    # Set the desired position
    x_des=np.array([0.6+5, 0, 0.6])

    # Set the desired velocities
    v_des=np.array([0, 0, 0])

    # Get the position error
    x_c = body.xpos 
    v_c = body.cvel[:3]

    # Get the velocity error
    delta_x=x_des-x_c
    delta_v=v_des-v_c

    # Get the Jacobian at the desired location on the robot
    # This function works by taking in return parameters!!! Make sure you supply it with placeholder
    # variables
    Jp=np.zeros((3,model.nv))
    Jr=np.zeros((3,model.nv))
    mj.mj_jacBody(model, data, Jp, Jr, body.id)

    # Compute the impedance control input torques
    Tau = Jp.T @ (K_d*delta_v + K_p*delta_x)
    
    # Set the control inputs
    data.ctrl[:7] = data.qfrc_bias[:7] + Tau[:7]

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    force[:] = np.roll(force, -1)[:]
    force[-1] = data.sensordata[2]

    # Update control inputs to the whiteboard
    get_board_control(model, data)

def position_control(model, data):

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    desired_joint_positions = np.array(
        [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (np.array([0, 0, 0, 0, 0, 0, 0])-data.qvel[:7])
    

####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)
    # print(model.nv)
    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF
    
    control_type= impedance_control # gravity_comp, position_control, force_control, impedance_control
    control_type_string = control_type.__name__
    mj.set_mjcb_control(control_type)  

    ################################# Swap Callback Above This Line #################################

    # Initialize variables to store force and time data points
    force_sensor_max_time = 10
    force = np.zeros(int(force_sensor_max_time/model.opt.timestep))
    time = np.linspace(0, force_sensor_max_time, int(
        force_sensor_max_time/model.opt.timestep))

    # Launch the simulate viewer
    with viewer.launch_passive(model, data) as v:
        for _ in range(int(force_sensor_max_time/model.opt.timestep)):
            mj.mj_step(model, data)
            v.sync()

    # Save recorded force and time points as a csv file
    force = np.reshape(force, (5000, 1))
    time = np.reshape(time, (5000, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt(f'moving_force_vs_time_{control_type_string}.csv', plot, delimiter=',')
