import time
from threading import Thread

import glfw
import mujoco
import numpy as np

class Mujoco_Sim_Env:

    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = [600.0, 600.0, 600.0, 30.0, 30.0, 30.0]
    height, width = 1000, 1200  # Rendering window resolution.
    fps = 30  # Rendering framerate.

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world.xml")
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 90.0
        self.cam.distance = 3.0
        self.cam.elevation = -45.0
        # self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.run = True
        self.gripper(True)
        self.gripper_state = "open"
        for i in range(1, 8):
            self.data.joint(f"panda_joint{i}").qpos = self.qpos0[i - 1]
        mujoco.mj_forward(self.model, self.data)

    def gripper(self, open=True):
        self.data.actuator("pos_panda_finger_joint1").ctrl = (0.04, 0)[not open]
        self.data.actuator("pos_panda_finger_joint2").ctrl = (0.04, 0)[not open]

    # def send_data(self, msg):
    
    #     msg = {'kinematic_twist': msg}
    #     pub.publish(msg)		                    # Send message each second
    #     time.sleep(0.001)

    # def control(self, xpos_d, xquat_d):
    #     xpos = self.data.body("panda_hand").xpos
    #     xquat = self.data.body("panda_hand").xquat
    #     jacp = np.zeros((3, self.model.nv))
    #     jacr = np.zeros((3, self.model.nv))
    #     bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
    #     mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

    #     error = np.zeros(6)
    #     error[:3] = xpos_d - xpos
    #     res = np.zeros(3)
    #     mujoco.mju_subQuat(res, xquat, xquat_d)
    #     mujoco.mju_rotVecQuat(res, res, xquat)
    #     error[3:] = -res

    #     J = np.concatenate((jacp, jacr))
    #     v = J @ self.data.qvel
    #     for i in range(1, 8):
    #         dofadr = self.model.joint(f"panda_joint{i}").dofadr
    #         self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
    #             f"panda_joint{i}"
    #         ).qfrc_bias
    #         self.data.actuator(f"panda_joint{i}").ctrl += (
    #             J[:, dofadr].T @ np.diag(self.K) @ error
    #         )
    #         self.data.actuator(f"panda_joint{i}").ctrl -= (
    #             J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
    #         )

    def control(self, xpos_d, xquat_d):
        xpos = self.data.body("panda_hand").xpos
        xquat = self.data.body("panda_hand").xquat
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos
        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        print(np.shape(J))
        for i in range(1, 8):
            dofadr = self.model.joint(f"panda_joint{i}").dofadr
            # self.data.actuator(f"panda_joint{i}").ctrl = self.data.joint(
            #     f"panda_joint{i}"
            # ).qfrc_bias
            # self.data.actuator(f"panda_joint{i}").ctrl += (
            #     J[:, dofadr].T @ np.diag(self.K) @ error
            # )
            # self.data.actuator(f"panda_joint{i}").ctrl -= (
            #     J[:, dofadr].T @ np.diag(2 * np.sqrt(self.K)) @ v
            # )

    def step(self) -> None:
        xpos0 = self.data.body("panda_hand").xpos.copy()
        xpos_d = xpos0
        xquat0 = self.data.body("panda_hand").xquat.copy()
        down = list(np.linspace(-0.65, 0, 2000))
        up = list(np.linspace(0, -0.65, 2000))
        state = "down"
        count = 0
        while self.run:
            if state == "down":
                if not count==len(down):
                    xpos_d = xpos0 + [0, 0, down[-count]]
                else:
                    state = "grasp"
                    count = 0
            elif state == "grasp":
                if self.gripper_state=="open":
                    self.gripper(False)
                    self.gripper_state = "close"
                else:
                    self.gripper(True)
                    self.gripper_state = "open"
                state = "up"
            elif state == "up":
                if not count==len(up):
                    xpos_d = xpos0 + [0, 0, up[-count]]
                else:
                    state = "down"
                    count = 0
            count+=1
            self.control(xpos_d, xquat0)
            mujoco.mj_step(self.model, self.data)
            time.sleep(1e-3)

    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Demo", None, None)
        glfw.make_context_current(window)
        self.context = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        while not glfw.window_should_close(window):
            w, h = glfw.get_framebuffer_size(window)
            viewport.width = w
            viewport.height = h
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scene,
            )
            mujoco.mjr_render(viewport, self.scene, self.context)
            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()
        self.run = False
        glfw.terminate()

    def start(self) -> None:
        step_thread = Thread(target=self.step)
        step_thread.start()
        self.render()