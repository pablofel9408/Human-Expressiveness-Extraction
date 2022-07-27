import utils
import pandas as pd
import numpy as np
import quaternion
import sys

def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[:,0], r[:,1], r[:,2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return np.column_stack((qw, qx, qy, qz))

def quaternion_multiply(quaternion1, quaternion0):
    q = np.copy(quaternion1)
    q[:,0] = -quaternion1[:,1] * quaternion0[:,1] - quaternion1[:,2] * quaternion0[:,2] - quaternion1[:,3] * quaternion0[:,3] + quaternion1[:,0] * quaternion0[:,1]
    q[:,1] = quaternion1[:,1] * quaternion0[:,0] + quaternion1[:,2] * quaternion0[:,3] - quaternion1[:,3] * quaternion0[:,2] + quaternion1[:,0] * quaternion0[:,1]
    q[:,2] = -quaternion1[:,1] * quaternion0[:,3] + quaternion1[:,2] * quaternion0[:,0] + quaternion1[:,3] * quaternion0[:,1] + quaternion1[:,0] * quaternion0[:,2]
    q[:,3] = quaternion1[:,1] * quaternion0[:,2] - quaternion1[:,2] * quaternion0[:,1] + quaternion1[:,3] * quaternion0[:,0] + quaternion1[:,0] * quaternion0[:,3]
    return q

def conjugate(q):
    q[:,1:] = -q[:,1:]
    return q

def rotate(a,q):

    # aa = np.empty((0,np.shape(a)[1]+1),dtype=np.float64)
    # for ele in zip(a,q):
    #     if np.round(np.linalg.norm(ele[1])) != 1.0:
    #         print("error")
    #         break
    a = np.column_stack((np.zeros((np.shape(a)[0],1)),a))
    print(np.shape(a))
    # a = np.insert(a,0,0)
    q0_conj = conjugate(q)
    print(np.shape(q0_conj))
    print(np.shape(q))
    v_r = quaternion_multiply(a,q0_conj)
    v_r = quaternion_multiply(q,v_r)
        # aa = np.append(aa, np.array([v_r]), axis=0)

    return v_r

def gravity_compensation(a,q):
    error = []
    gravity = np.ones((np.shape(a)[0],np.shape(a)[1]+1))*[0,0, 0, -9.81]
    a_rotated = rotate(a, q)
    print(np.max(a[:,2]))
    print(np.max(a_rotated[:,3]))
    user_acceleration = a_rotated - gravity
    print(np.max(user_acceleration[:,3]))

cols_oi = ["v_x", "v_y", "v_z", "v_ang_x", "v_ang_y", "v_ang_z"]
cols_oi_1 = ["roll", "pitch", "yaw"]
a = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base\\Sequence 1\\ms25.csv", header=None)
a = a[[4,5,6,7,8,9]]
a.columns = cols_oi

a1 = pd.read_csv("C:\\Users\\posorio\\Documents\\Expressive movement\\Data Engineering\\Datasets\\Robot\\Mobile base\\Sequence 1\\ms25_euler.csv", header=None)
a1 = a1[[1,2,3]]
a1.columns = cols_oi_1
a1 = a1.to_numpy()
a = a.to_numpy()
print(np.shape(a1)[0])
print(np.shape(a)[0])

q = euler_to_quaternion(a1)
gravity_compensation(a[:np.shape(q)[0],:3],q)