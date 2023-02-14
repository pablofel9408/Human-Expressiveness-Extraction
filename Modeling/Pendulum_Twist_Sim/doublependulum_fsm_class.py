import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import csv
import pandas as pd
import time

from .constanst import *

class FSM_Sim():
    def __init__(self) -> None:
        
        # For callback functions
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # Define setpoint times
        self.t_init = t_hold
        self.t_mid = t_hold + t_swing1
        self.t_end = t_hold + t_swing1 + t_swing2

        self.q_init = np.array([[-1.0], [0.0]])
        self.q_mid = np.array([[1.0], [0.0]])
        self.q_end = np.array([[-1.0], [0.0]])

        if SAVE_TRAJECTORY:
            self.file = open('pendulum_twist.csv', 'a')
            self.writer = csv.writer(self.file)

        if LOAD_TRAJECTORY:
            self.count = 0

    def get_xml_path(self):
        #get the full path
        xml_path = 'doublependulum_fsm_multi.xml'
        dirname = os.path.dirname(__file__)
        abspath = os.path.join(dirname + "/" + xml_path)
        xml_path = abspath

        return xml_path

    def keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)

    def mouse_button(self, window, button, act, mods):
        # update button state
        button_left = (glfw.get_mouse_button(
            self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        button_middle = (glfw.get_mouse_button(
            self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        button_right = (glfw.get_mouse_button(
            self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

        # update mouse position
        glfw.get_cursor_pos(self.window)

    def mouse_move(self, window, xpos, ypos):
        # compute mouse displacement, save
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        # no buttons down: nothing to do
        if (not self.button_left) and (not self.button_middle) and (not self.button_right):
            return

        # get current window size
        width, height = glfw.get_window_size(self.window)

        # get shift key state
        PRESS_LEFT_SHIFT = glfw.get_key(
            window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        PRESS_RIGHT_SHIFT = glfw.get_key(
            window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

        # determine action based on mouse button
        if self.button_right:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_MOVE_H
            else:
                action = mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            if mod_shift:
                action = mj.mjtMouse.mjMOUSE_ROTATE_H
            else:
                action = mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height,
                        dy/height, self.scene, self.cam)

    def scroll(self,window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                        yoffset, self.scene, self.cam)
    
    def set_basic_attirbutes(self):

        # MuJoCo data structures
        self.model = mj.MjModel.from_xml_path(self.get_xml_path())  # MuJoCo model
        self.data = mj.MjData(self.model)                # MuJoCo data
        self.cam = mj.MjvCamera()                        # Abstract camera
        self.opt = mj.MjvOption()                        # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        self.window = glfw.create_window(1200, 900, "Demo", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # install GLFW mouse and keyboard callbacks
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_scroll_callback(self.window, self.scroll)
   
    def set_initial_conditions(self):
        # Set initial condition
        self.data.qpos[0] = -1
        self.data.qpos[2] = -1

    def set_cam_initila_conditions(self):

        # Set camera configuration
        self.cam.azimuth = 89.608063
        self.cam.elevation = -11.588379
        self.cam.distance = 5.0
        self.cam.lookat = np.array([0.0, 0.0, 1.5])

    def generate_trajectory(self, t0, tf, q0, qf):
        """
        Generates a trajectory
        q(t) = a0 + a1t + a2t^2 + a3t^3
        which satisfies the boundary condition
        q(t0) = q0, q(tf) = qf, dq(t0) = 0, dq(tf) = 0
        """
        tf_t0_3 = (tf - t0)**3
        a0 = qf*(t0**2)*(3*tf-t0) + q0*(tf**2)*(tf-3*t0)
        a0 = a0 / tf_t0_3

        a1 = 6 * t0 * tf * (q0 - qf)
        a1 = a1 / tf_t0_3

        a2 = 3 * (t0 + tf) * (qf - q0)
        a2 = a2 / tf_t0_3

        a3 = 2 * (q0 - qf)
        a3 = a3 / tf_t0_3

        return a0, a1, a2, a3
    
    def init_controller(self):

        self.fsm_state = FSM_HOLD

        self.a_swing1 = self.generate_trajectory(self.t_init, self.t_mid, self.q_init, self.q_mid)

        self.a_swing2 = self.generate_trajectory(self.t_mid, self.t_end, self.q_mid, self.q_end)

    def load_trajectories(self, num_traj):
        if num_traj==1:
            trajectory = pd.read_csv(pendulum_trajectory_filepath)
            trajectory = trajectory.to_numpy()
        elif num_traj==2:
            trajectory = np.load(trajectory_network_same_filepath)
        elif num_traj==3:
            trajectory = np.load(trajectory_network_random_filepath)
        elif num_traj==4:
            # trajectory = np.load("C:\\Users\\posorio\\Downloads\\5_dbpendulum_fsm\\5_dbpendulum_fsm\human_same_input_samples.npy")
            # trajectory = trajectory/500
            trajectory = np.load(trajectory_human_same_filepath_vel)
            # trajectory = trajectory/10
        elif num_traj==5:
            trajectory = np.load(trajectory_human_random_filepath_vel)
            # trajectory = trajectory/500
            # trajectory[:,2] = trajectory[:,2]/2
        print(np.shape(trajectory))

        return trajectory

    def load_trajectory_from_path(self, path):
        return np.load(path)

    def set_trajectory(self, from_path=False, paths=[]):

        if not from_path:
            self.trajectory_1 = self.load_trajectories(num_traj=1)
            self.trajectory_2 = self.load_trajectories(num_traj=flag_cst)
        else:
            self.trajectory_1 = self.load_trajectory_from_path(paths[0])
            self.trajectory_2 = self.load_trajectory_from_path(paths[1])\

    def controller(self, model,data):
        """
        This function implements a PD controller for tracking
        the reference motion.
        """
        time = self.data.time
        
        # Check for state change
        if self.fsm_state == FSM_HOLD and time >= t_hold:
            self.fsm_state = FSM_SWING1
        elif self.fsm_state == FSM_SWING1 and time >= self.t_mid:
            self.fsm_state = FSM_SWING2
        elif self.fsm_state == FSM_SWING2 and time >= self.t_end:
            self.fsm_state = FSM_SWING1

        for i in range(0,6):
            self.data.ctrl[i]=0

        # End-effector position
        # end_eff_pos = data.sensordata[:3]
        # print(data.sensordata)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacBody(self.model, self.data, jacp, jacr, 3)
        J = np.concatenate((jacp, jacr))

        if LOAD_TRAJECTORY:
            dq_ref = np.linalg.pinv(J) @ self.trajectory_1[self.count]
            self.data.ctrl[1] = self.data.qpos[0] + dq_ref[0]
            self.data.ctrl[4] = self.data.qpos[1] + dq_ref[1]

            dq_ref = np.linalg.pinv(J) @ self.trajectory_2[self.count]
            self.data.ctrl[7] = self.data.qpos[0] + dq_ref[0]
            self.data.ctrl[10] = self.data.qpos[1] + dq_ref[1]
            self.count+=1

        else:
            q_vel = self.data.qvel
            v = J @ q_vel

            if SAVE_TRAJECTORY:
                self.writer.writerow(v)

            # Get reference joint position & velocity
            if self.fsm_state == FSM_HOLD:
                q_ref = self.q_init
                dq_ref = np.zeros((2, 1))
            elif self.fsm_state == FSM_SWING1:
                q_ref = self.a_swing1[0] + self.a_swing1[1]*time + \
                    self.a_swing1[2]*(time**2) + self.a_swing1[3]*(time**3)
                dq_ref = self.a_swing1[1] + 2 * self.a_swing1[2] * \
                    time + 3 * self.a_swing1[3]*(time**2)
            elif self.fsm_state == FSM_SWING2:
                q_ref = self.a_swing2[0] + self.a_swing2[1]*time + \
                    self.a_swing2[2]*(time**2) + self.a_swing2[3]*(time**3)
                dq_ref = self.a_swing2[1] + 2 * self.a_swing2[2] * \
                    time + 3 * self.a_swing2[3]*(time**2)
            elif self.fsm_state == FSM_STOP:
                q_ref = self.q_end
                dq_ref = np.zeros((2, 1))

            # Define PD gains
            kp = 500
            kv = 50

            # Compute PD control
            torque = kp * (q_ref[:, 0] - self.data.qpos) + \
                 kv * (dq_ref[:, 0] - self.data.qvel)

            self.data.ctrl[0] = torque[0]
            self.data.ctrl[3] = torque[1]
        
    def sim_loop(self):
        
        mj.set_mjcb_control(self.controller)
        aa = time.time()
        nn = 0
        while not glfw.window_should_close(self.window):
            simstart = self.data.time

            while (self.data.time - simstart < 1.0/60.0):
                    mj.mj_step(self.model, self.data)

            # if (data.time>=simend):
            #     break;
            # print(count)

            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                self.window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                            mj.mjtCatBit.mjCAT_ALL.value, self.scene)
            mj.mjr_render(viewport, self.scene, self.context)

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(self.window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

            # print(nn)
            # print(count)
            if nn < 1:
                time.sleep(5)
                nn = 3

        if SAVE_TRAJECTORY:
            self.file.close()

        glfw.terminate()

    def start_simulation(self):

        self.set_basic_attirbutes()
        self.set_cam_initila_conditions()
        self.set_initial_conditions()
        self.init_controller()

        if LOAD_TRAJECTORY:   
            self.set_trajectory(from_path=LOAD_FROM_ARCHIVE,paths=path_to_compare)

        self.sim_loop()