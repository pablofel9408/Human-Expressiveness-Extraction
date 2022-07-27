import numpy as np 
import os
import matplotlib.pyplot as plt

support_dir = '//mnt/c/Users/posorio/Documents/Expressive movement/Data Engineering/Datasets/Human/Dance RB/DanceDB/DanceDB'
amass_npz_fname = os.path.join(support_dir, '20130216_AnnaCharalambous/Anna_Angry_C3D_poses.npz') # the path to body data
bdata = np.load(amass_npz_fname)

# you can set the gender manually and if it differs from data's then contact or interpenetration issues might happen
subject_gender = bdata['gender']

print('Data keys available:%s'%list(bdata.keys()))

print('The subject of the mocap sequence is  {}.'.format(subject_gender))
print('The subject of the mocap frame rate is  {}.'.format(bdata['mocap_framerate']))
print(np.shape(bdata['poses']))
print(np.shape(bdata['dmpls']))
plt.plot(bdata['trans'])
plt.show()

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import torch

imw, imh=1600, 1600
# mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
time_length = len(bdata['trans'])
# Choose the device to run the body model on.
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_betas = 16 # number of body parameters
num_dmpls = 8 # number of DMPL parameters

bm = BodyModel(bm_fname=amass_npz_fname, num_betas=num_betas,).to(comp_device)

body_parms = {
    'root_orient': torch.Tensor(bdata['poses'][:, :3]).to(comp_device), # controls the global root orientation
    'pose_body': torch.Tensor(bdata['poses'][:, 3:66]).to(comp_device), # controls the body
    'pose_hand': torch.Tensor(bdata['poses'][:, 66:]).to(comp_device), # controls the finger articulation
    'trans': torch.Tensor(bdata['trans']).to(comp_device), # controls the global body position
    'betas': torch.Tensor(np.repeat(bdata['betas'][:num_betas][np.newaxis], repeats=time_length, axis=0)).to(comp_device), # controls the body shape. Body shape is static
    'dmpls': torch.Tensor(bdata['dmpls'][:, :num_dmpls]).to(comp_device) # controls soft tissue dynamics
}
body_pose_hand = bm(**{k:v for k,v in body_parms.items() if k in ['pose_body', 'betas', 'pose_hand']})

def vis_body_joints(fId = 0):
    joints = c2c(body_pose_hand.Jtr[fId])
    joints_mesh = points_to_spheres(joints, point_color = colors['red'], radius=0.005)

    # mv.set_static_meshes([joints_mesh])
    # body_image = mv.render(render_wireframe=False)
    # show_image(body_image)

vis_body_joints(fId=0)
