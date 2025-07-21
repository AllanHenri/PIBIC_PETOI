import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time


# Constants to define training and visualisation.
GUI_MODE = False          # Set "True" to display pybullet in a window
EPISODE_LENGTH = 250      # Number of steps for one training episode
MAXIMUM_LENGTH = 1.8e6    # Number of total steps for entire training

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6       # Increase of penalty by step_counter/PENALTY_STEPS
FAC_MOVEMENT = 1000       # Reward movement in x-direction
FAC_STABILITY = 0.1       # Punish body roll and pitch velocities
FAC_Z_VELOCITY = 0.0      # Punish z movement of body
FAC_SLIP = 0.0            # Punish slipping of paws
FAC_ARM_CONTACT = 0.01    # Punish crawling on arms and elbows
FAC_SMOOTH_1 = 1.0        # Punish jitter and vibrational movement, 1st order
FAC_SMOOTH_2 = 1.0        # Punish jitter and vibrational movement, 2nd order
FAC_CLEARANCE = 0.0       # Factor to enfore foot clearance to PAW_Z_TARGET
PAW_Z_TARGET = 0.005      # Target height (m) of paw during swing phase

BOUND_ANG = 110         # Joint maximum angle (deg)
STEP_ANGLE = 11           # Maximum angle (deg) delta per step
ANG_FACTOR = 0.1          # Improve angular velocity resolution before clip.

# Values for randomization, to improve sim to real transfer.
RANDOM_GYRO = 0           # Percent
RANDOM_JOINT_ANGS = 0      # Percent
RANDOM_MASS = 0           # Percent, currently inactive
RANDOM_FRICTION = 0       # Percent, currently inactive

LENGTH_RECENT_ANGLES = 3  # Buffer to read recent joint angles
LENGTH_JOINT_HISTORY = 30 # Number of steps to store joint angles.

# Size of oberservation space is set up of: 
# [LENGTH_JOINT_HISTORY, quaternion, gyro]
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6     


class OpenCatGymEnv(gym.Env):
    """ Gymnasium environment (stable baselines 3) for OpenCat robots.
    """

    metadata = {'render.modes': ['human']}

class OpenCatGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

class OpenCatGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_history = np.array([])
        self.angle_history = np.array([])
        self.bound_ang = np.deg2rad(BOUND_ANG)

        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5,
                                     cameraYaw=-170,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.4,0,0])

        # Apenas 6 motores (sem a pata dianteira esquerda: índices 0 e 1)
        self.joints_to_control = [0, 1, 2, 3, 4, 5, 6, 7]
        self.action_space = gym.spaces.Box(np.array([-1]*6), np.array([1]*6), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.array([-1]*SIZE_OBSERVATION),
                                                np.array([1]*SIZE_OBSERVATION), dtype=np.float32)


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:, 0]
        ds = np.deg2rad(STEP_ANGLE)

        # Aplica ação apenas nas juntas que controlamos (6 motores)
        for i, j_idx in enumerate(self.joints_to_control):
            joint_angs[j_idx] += action[i] * ds
            joint_angs[j_idx] = np.clip(joint_angs[j_idx], -self.bound_ang, self.bound_ang)

        # Trava as juntas da pata removida
        joint_angs[0] = 0  # shoulder_left
        joint_angs[1] = 0  # elbow_left

        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64)).round()
        joint_angs = np.deg2rad(joint_angsDeg)

        p.setJointMotorControlArray(self.robot_id,
                                    self.joint_id,
                                    p.POSITION_CONTROL,
                                    joint_angs,
                                    forces=np.ones(8) * 0.2)
        p.stepSimulation()

        # Ignora a pata dianteira esquerda no contato e clearance
        paw_idx = [ 3, 6, 9, 12]  # Sem a 3 (dianteira esquerda)
        paw_slipping = 0
        paw_clearance = 0
        for idx in paw_idx:
            contacts = p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx)
            if contacts:
                vel = p.getLinkState(self.robot_id, linkIndex=idx, computeLinkVelocity=1)[0][0:1]
                paw_slipping += np.linalg.norm(vel)
                paw_z_pos = p.getLinkState(self.robot_id, linkIndex=idx)[0][2]
                paw_clearance += (paw_z_pos - PAW_Z_TARGET)**2 * np.linalg.norm(vel)**0.5

        # Ignora braço esquerdo nos contatos
        arm_idx = [0, 2, 4, 5]  # Sem os índices 0 e 1
        for idx in arm_idx:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx):
                self.arm_contact += 1


        # Read clearance of torso from ground
        base_clearance = p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # Set new joint angles
        p.setJointMotorControlArray(self.robot_id, 
                                    self.joint_id, 
                                    p.POSITION_CONTROL, 
                                    joint_angs, 
                                    forces=np.ones(8)*0.2)
        p.stepSimulation() # Delay of data transfer

        # Normalize joint_angs
        joint_angs[0] /= self.bound_ang
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Adding every 2nd angle to the joint angle history.
        if(self.step_counter % 2 == 0):
            self.angle_history = np.append(self.angle_history, 
                                           self.randomize(joint_angs, 
                                                          RANDOM_JOINT_ANGS))
            self.angle_history = np.delete(self.angle_history, np.s_[0:8])

        self.recent_angles = np.append(self.recent_angles, joint_angs)
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]
        joint_angs_prev_prev = self.recent_angles[0:8]

        # Read robot state (pitch, roll and their derivatives of the torso).
        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        p.stepSimulation() # Emulated delay of data transfer via serial port
        state_ang_euler = np.asarray(p.getEulerFromQuaternion(state_ang)[0:2])
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        state_vel_clip = np.clip(state_vel, -1, 1)
        self.state_robot = np.concatenate((state_ang, state_vel_clip))
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0] 

        # Penalty and reward
        smooth_movement = np.sum(
            FAC_SMOOTH_1*np.abs(joint_angs-joint_angs_prev)**2
            + FAC_SMOOTH_2*np.abs(joint_angs
            - 2*joint_angs_prev 
            + joint_angs_prev_prev)**2)

        z_velocity = p.getBaseVelocity(self.robot_id)[0][2]

        body_stability = (FAC_STABILITY * (state_vel_clip[0]**2 
                                          + state_vel_clip[1]**2) 
                                          + FAC_Z_VELOCITY * z_velocity**2)

        movement_forward = current_position - last_position
        reward = (FAC_MOVEMENT * movement_forward 
                 - self.step_counter_session/PENALTY_STEPS * (
                    smooth_movement + body_stability 
                    + FAC_CLEARANCE * paw_clearance 
                    + FAC_SLIP * paw_slipping**2 
                    + FAC_ARM_CONTACT * self.arm_contact))

        # Set state of the current state.
        terminated = False
        truncated = False
        info = {}

        # Stop criteria of current learning episode: 
        # Number of steps or robot fell.
        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            terminated = False
            truncated = True

        elif self.is_fallen(): # Robot fell
            self.step_counter_session += self.step_counter
            reward = 0
            terminated = True
            truncated = False

        self.observation = np.hstack((self.state_robot, self.angle_history))

        return (np.array(self.observation).astype(np.float32), reward, terminated, truncated, info)


    def reset(self, seed=None, options=None):
        self.step_counter = 0
        self.arm_contact = 0
        p.resetSimulation()
        # Disable rendering during loading.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) 
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        urdf_path = "models/"#"/content/drive/My Drive/opencat-gym-esp32/models/"
        self.robot_id = p.loadURDF(urdf_path + "bittle_esp32.urdf", 
                                   start_pos, start_orient, 
                                   flags=p.URDF_USE_SELF_COLLISION) 
        
        # Initialize urdf links and joints.
        self.joint_id = []
        #paramIds = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            joint_name = info[1]
            joint_type = info[2]

            if (joint_type == p.JOINT_PRISMATIC 
                or joint_type == p.JOINT_REVOLUTE):
                self.joint_id.append(j)
                #paramIds.append(p.addUserDebugParameter(joint_name.decode("utf-8")))
                # Limiting motor dynamics. Although bittle's dynamics seem to 
                # be be quite high like up to 7 rad/s.
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10) 
        
        # Setting start position. This influences training.
        joint_angs = np.deg2rad(np.array([1, 0, 1, 0, 1, 0, 1, 0])*50) 

        i = 0
        for j in self.joint_id:
            p.resetJointState(self.robot_id,j, joint_angs[i])
            i = i+1

        # Normalize joint angles.
        joint_angs[0] /= self.bound_ang
        joint_angs[1] /= self.bound_ang
        joint_angs[2] /= self.bound_ang
        joint_angs[3] /= self.bound_ang
        joint_angs[4] /= self.bound_ang
        joint_angs[5] /= self.bound_ang
        joint_angs[6] /= self.bound_ang
        joint_angs[7] /= self.bound_ang

        # Read robot state (pitch, roll and their derivatives of the torso)
        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        self.state_robot = np.concatenate((state_ang, 
                                           np.clip(state_vel, -1, 1)))

        # Initialize robot state history with reset position
        state_joints = np.asarray(
            p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        state_joints /= self.bound_ang 
        
        self.angle_history = np.tile(state_joints, LENGTH_JOINT_HISTORY)
        self.recent_angles = np.tile(state_joints, LENGTH_RECENT_ANGLES)
        self.observation = np.concatenate((self.state_robot, 
                                           self.angle_history))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return np.array(self.observation).astype(np.float32), info


    def render(self, mode='human'):
        if self.render_mode == "human":
            time.sleep(1. / 60)


    def close(self):
        p.disconnect()


    def is_fallen(self):
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        return (np.fabs(orient[0]) > 1.3 or np.fabs(orient[1]) > 1.3)


    def randomize(self, value, percentage):
        percentage /= 100
        return value * (1 + percentage * (2 * np.random.rand() - 1))
