"""Demonstrates the Franka Emika Robot System model for MuJoCo."""

import time
from threading import Thread

import glfw
import mujoco
import numpy as np

import nep                                # Add nep library
import time                               # Add time library

from sim_env import Mujoco_Sim_Env

# node_pub = nep.node('send_robot_movement')                 # Define node name
# pub = node_pub.new_pub('robot_movement','json') 

# node_rec = nep.node("receive_network_output")             # Create a new nep node
# sub = node_rec.new_sub("test_topic", "json")    # Set the topic and message type

# def read_data():

#     while True:
#         is_message, msg = sub.listen()    # Non blocking socket
#         if is_message:                    # Message avaliable only if is_message == True
#             # Here put your code for processing the message
#             # ...
#             print (msg)
#         else:
#             time.sleep(.0001)

def main():

    panda_env = Mujoco_Sim_Env()
    panda_env.start()


if __name__ == "__main__":
    main()