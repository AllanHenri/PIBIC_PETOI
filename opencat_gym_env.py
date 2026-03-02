import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data

# Constants to define training and visualisation.
GUI_MODE = False
EPISODE_LENGTH = 250
MAXIMUM_LENGTH = 1.8e6

# Factors to weight rewards and penalties.
PENALTY_STEPS = 2e6
FAC_MOVEMENT = 1000
FAC_STABILITY = 0.1
FAC_Z_VELOCITY = 0.0
FAC_SLIP = 0.0
FAC_ARM_CONTACT = 0.01
FAC_SMOOTH_1 = 1.0
FAC_SMOOTH_2 = 1.0
FAC_CLEARANCE = 0.0
PAW_Z_TARGET = 0.005

BOUND_ANG = 110
STEP_ANGLE = 11
ANG_FACTOR = 0.1

RANDOM_GYRO = 0
RANDOM_JOINT_ANGS = 0
RANDOM_MASS = 0
RANDOM_FRICTION = 0

LENGTH_RECENT_ANGLES = 3
LENGTH_JOINT_HISTORY = 30
SIZE_OBSERVATION = LENGTH_JOINT_HISTORY * 8 + 6

class OpenCatGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        self.step_counter_session = 0
        self.state_history = np.array([])
        self.angle_history = np.array([])
        self.bound_ang = np.deg2rad(BOUND_ANG)

        if GUI_MODE:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-170, cameraPitch=-40, cameraTargetPosition=[0.4,0,0])

        self.action_space = gym.spaces.Box(np.array([-1]*8), np.array([1]*8))
        self.observation_space = gym.spaces.Box(np.array([-1]*SIZE_OBSERVATION), np.array([1]*SIZE_OBSERVATION))

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        last_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]
        joint_angs = np.asarray(p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        ds = np.deg2rad(STEP_ANGLE)

        # === TRAVAR PATA DIANTEIRA ESQUERDA ===
        locked_shoulder_angle = np.deg2rad(30)
        locked_elbow_angle = np.deg2rad(45)
        joint_angs[0] = locked_shoulder_angle
        joint_angs[1] = locked_elbow_angle
        action[0] = 0
        action[1] = 0

        joint_angs += action * ds

        min_ang = -self.bound_ang
        max_ang = self.bound_ang
        for i in range(8):
            joint_angs[i] = np.clip(joint_angs[i], min_ang, max_ang)

        joint_angsDeg = np.rad2deg(joint_angs.astype(np.float64))
        joint_angsDegRounded = joint_angsDeg.round()
        joint_angs = np.deg2rad(joint_angsDegRounded)

        p.stepSimulation()

        paw_contact = []
        paw_idx = [3, 6, 9, 12]
        for idx in paw_idx:
            paw_contact.append(True if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx) else False)

        paw_slipping = 0
        for in_contact in np.nonzero(paw_contact)[0]:
            link_state = p.getLinkState(self.robot_id, linkIndex=paw_idx[in_contact], computeLinkVelocity=1)
            paw_linear_velocity_xy = np.asarray(link_state[6][0:2])
            paw_slipping += np.linalg.norm(paw_linear_velocity_xy)

        paw_clearance = 0
        for idx in paw_idx:
            link_state = p.getLinkState(self.robot_id, linkIndex=idx, computeLinkVelocity=1)
            paw_z_pos = link_state[0][2]
            paw_linear_velocity_xy = np.asarray(link_state[6][0:2])
            paw_clearance += (paw_z_pos-PAW_Z_TARGET)**2 * np.linalg.norm(paw_linear_velocity_xy)**0.5

        arm_idx = [1, 2, 4, 5]
        for idx in arm_idx:
            if p.getContactPoints(bodyA=self.robot_id, linkIndexA=idx):
                self.arm_contact += 1

        base_clearance = p.getBasePositionAndOrientation(self.robot_id)[0][2]

        p.setJointMotorControlArray(self.robot_id, self.joint_id, p.POSITION_CONTROL, joint_angs, forces=np.ones(8)*0.2)
        p.stepSimulation()

        for i in range(8):
            joint_angs[i] /= self.bound_ang

        if(self.step_counter % 2 == 0):
            self.angle_history = np.append(self.angle_history, self.randomize(joint_angs, RANDOM_JOINT_ANGS))
            self.angle_history = np.delete(self.angle_history, np.s_[0:8])

        self.recent_angles = np.append(self.recent_angles, joint_angs)
        self.recent_angles = np.delete(self.recent_angles, np.s_[0:8])

        joint_angs_prev = self.recent_angles[8:16]
        joint_angs_prev_prev = self.recent_angles[0:8]

        state_pos, state_ang = p.getBasePositionAndOrientation(self.robot_id)
        p.stepSimulation()
        state_ang_euler = np.asarray(p.getEulerFromQuaternion(state_ang)[0:2])
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        state_vel_clip = np.clip(state_vel, -1, 1)
        self.state_robot = np.concatenate((state_ang, state_vel_clip))
        current_position = p.getBasePositionAndOrientation(self.robot_id)[0][0]

        smooth_movement = np.sum(FAC_SMOOTH_1*np.abs(joint_angs-joint_angs_prev)**2 + FAC_SMOOTH_2*np.abs(joint_angs - 2*joint_angs_prev + joint_angs_prev_prev)**2)
        z_velocity = p.getBaseVelocity(self.robot_id)[0][2]
        body_stability = (FAC_STABILITY * (state_vel_clip[0]**2 + state_vel_clip[1]**2) + FAC_Z_VELOCITY * z_velocity**2)
        movement_forward = current_position - last_position
        reward = (FAC_MOVEMENT * movement_forward - self.step_counter_session/PENALTY_STEPS * (smooth_movement + body_stability + FAC_CLEARANCE * paw_clearance + FAC_SLIP * paw_slipping**2 + FAC_ARM_CONTACT * self.arm_contact))

        terminated = False
        truncated = False
        info = {}

        self.step_counter += 1
        if self.step_counter > EPISODE_LENGTH:
            self.step_counter_session += self.step_counter
            terminated = False
            truncated = True
        elif self.is_fallen():
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
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        p.setGravity(0,0,-9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        start_pos = [0,0,0.08]
        start_orient = p.getQuaternionFromEuler([0,0,0])

        urdf_path = "models/"
        self.robot_id = p.loadURDF(urdf_path + "bittle_esp32.urdf", start_pos, start_orient, flags=p.URDF_USE_SELF_COLLISION)

        self.joint_id = []
        for j in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, j)
            joint_name = info[1]
            joint_type = info[2]
            if (joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE):
                self.joint_id.append(j)
                p.changeDynamics(self.robot_id, j, maxJointVelocity = np.pi*10)

        joint_angs = np.array([
            np.deg2rad(45),  # shoulder_left
            np.deg2rad(0),  # elbow_left
            np.deg2rad(50),
            np.deg2rad(0),
            np.deg2rad(50),
            np.deg2rad(0),
            np.deg2rad(50),
            np.deg2rad(0),
        ])

        for i, j in enumerate(self.joint_id):
            p.resetJointState(self.robot_id,j, joint_angs[i])

        p.setJointMotorControl2(self.robot_id, self.joint_id[0], p.POSITION_CONTROL, targetPosition=np.deg2rad(30), force=0.3)
        p.setJointMotorControl2(self.robot_id, self.joint_id[1], p.POSITION_CONTROL, targetPosition=np.deg2rad(45), force=0.3)

        for i in range(8):
            joint_angs[i] /= self.bound_ang

        state_ang = p.getBasePositionAndOrientation(self.robot_id)[1]
        state_vel = np.asarray(p.getBaseVelocity(self.robot_id)[1])
        state_vel = state_vel[0:2]*ANG_FACTOR
        self.state_robot = np.concatenate((state_ang, np.clip(state_vel, -1, 1)))

        state_joints = np.asarray(p.getJointStates(self.robot_id, self.joint_id), dtype=object)[:,0]
        state_joints /= self.bound_ang
        self.angle_history = np.tile(state_joints, LENGTH_JOINT_HISTORY)
        self.recent_angles = np.tile(state_joints, LENGTH_RECENT_ANGLES)
        self.observation = np.concatenate((self.state_robot, self.angle_history))
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        info = {}
        return np.array(self.observation).astype(np.float32), info

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

    def is_fallen(self):
        pos, orient = p.getBasePositionAndOrientation(self.robot_id)
        orient = p.getEulerFromQuaternion(orient)
        is_fallen = (np.fabs(orient[0]) > 1.3 or np.fabs(orient[1]) > 1.3)
        return is_fallen

    def randomize(self, value, percentage):
        percentage /= 100
        value_randomized = value * (1 + percentage*(2*np.random.rand()-1))
        return value_randomized
