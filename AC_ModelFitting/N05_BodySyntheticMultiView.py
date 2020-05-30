#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


SMPLSH_Dir = r'..\SMPL_Socks\SMPL_reimp'

import sys
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch
import numpy as np

import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from tqdm import tqdm_notebook
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,

    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    TexturedSoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    BlendParams
)
from pytorch3d.transforms.so3 import (
    so3_exponential_map,
    so3_relative_angle,
)
# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))
import json
from os.path import join


# In[3]:


outFolder = r'F:\WorkingCopy2\2020_04_20_DifferentiableRendererTest\MultiView'


# In[4]:


smplshData = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SmplshModel.npz'
# Setup
device = torch.device("cuda:0")
torch.cuda.set_device(device)

pose_size = 3 * 52
beta_size = 10

smplsh = smplsh_torch.SMPLModel(device, smplshData)
np.random.seed(9608)
pose = torch.from_numpy((np.random.rand(pose_size) - 0.5) * 0.4)        .type(torch.float64).to(device)
betas = torch.from_numpy((np.random.rand(beta_size) - 0.5) * 0.06)         .type(torch.float64).to(device)
trans = torch.from_numpy(np.zeros(3)).type(torch.float64).to(device)

verts = smplsh(betas, pose, trans).type(torch.float32)
# Initialize each vertex to be gray in color.
verts_rgb = ( 0.5 *torch.ones_like(verts))[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))

smplshMesh = Meshes([verts], [smplsh.faces.to(device)], textures=textures)


# In[6]:


class RenderingCfg:
    def __init__(s):
        s.sigma = 1e-4
        s.blurRange = 1e-4
        s.faces_per_pixel = 50
        s.bodyJointOnly = False
        s.randSeedPerturb = 1234
        s.noiseLevel = 0.5
        s.numIterations = 2000
        s.learningRate = 0.005
        s.terminateLoss = 200
        s.plotStep = 10
        s.numCams = 8
        s.imgSize = 1024
        
cfg = RenderingCfg()
cfg.terminateLoss = 100
# cfg.faces_per_pixel = 20
# cfg.blurRange = 0
# cfg.blurRange = 1e-3
# cfg.sigma = 1e-5
# cfg.numCams = 16


# In[7]:


expName = 'Param_Sig' + str(cfg.sigma) + '_BRg' + str(cfg.blurRange) + '_Fpp' + str(cfg.faces_per_pixel) \
          + '_BO' + str(cfg.bodyJointOnly) + '_NCams' + str(cfg.numCams) + '_IS' + str(cfg.imgSize)
outFolderForExperiment = join(outFolder, expName)
os.makedirs(outFolderForExperiment, exist_ok=True)
print(outFolderForExperiment)

json.dump(cfg.__dict__, open(join(outFolderForExperiment, 'cfg.json'), 'w'), indent=2)


# In[8]:


camRTs = []
for iCam in range(cfg.numCams):
    R, T = look_at_view_transform(2.7, 0, 360 * iCam / cfg.numCams, device=device) 
    camRTs.append({'R':R, 'T':T})


# In[9]:


print(camRTs[0])


# In[10]:


# Initialize an OpenGL perspective camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
cameras = OpenGLPerspectiveCameras(device=device, R=camRTs[0]['R'], T=camRTs[0]['T'])

# blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
blend_params = BlendParams(sigma=cfg.sigma, gamma=1e-4)


# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to rasterize_meshes.py for explanations of these parameters. 
if cfg.blurRange!= 0:
    raster_settings = RasterizationSettings(
        image_size=cfg.imgSize,
        blur_radius= np.log(1. / cfg.blurRange - 1.) * blend_params.sigma, 
        faces_per_pixel=cfg.faces_per_pixel, 
        bin_size=0
    )
else:
    raster_settings = RasterizationSettings(
        image_size=cfg.imgSize,
        blur_radius= 0, 
        faces_per_pixel=cfg.faces_per_pixel, 
        bin_size=0
    )

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )
renderer = MeshRenderer(
    rasterizer = rasterizer,
#     shader=SoftPhongShader(
#         device=device, 
#         cameras=cameras,
#         lights=lights,
#         blend_params=blend_params
#     )
    shader=SoftSilhouetteShader(
        blend_params=blend_params
        # device=device, 
        # cameras=cameras,
        # lights=lights
    )
)


# In[11]:


refImgs = []
fig, axs = plt.subplots(1, cfg.numCams)
fig.set_size_inches(cfg.numCams*2, 2)
with torch.no_grad():
    for iCam in range(cfg.numCams):
        R=camRTs[iCam]['R']
        T=camRTs[iCam]['T']
        image = renderer(meshes_world=smplshMesh, R=R, T=T)
        axs[iCam].imshow(image[0,...,3].cpu().numpy())
        refImgs.append(image)
        axs[iCam].axis('off')
outTargetImgFile = join(outFolderForExperiment, 'TargetImg.png')
fig.savefig(outTargetImgFile, dpi=cfg.imgSize, transparent=True, bbox_inches='tight', pad_inches=0)
outTargetImgFilePdf = join(outFolderForExperiment, 'TargetImg.pdf')
fig.savefig(outTargetImgFilePdf, dpi=cfg.imgSize, transparent=True, bbox_inches='tight', pad_inches=0)


# In[12]:


memStats = torch.cuda.memory_stats(device=device)
print('Before release: active_bytes.all.current:', memStats['active_bytes.all.current'] / 1000000)
torch.cuda.empty_cache()
memStats = torch.cuda.memory_stats(device=device)
print('After release: active_bytes.all.current:', memStats['active_bytes.all.current'] / 1000000)


# In[13]:


np.random.seed(cfg.randSeedPerturb)

if cfg.bodyJointOnly:
    numParameters = 3 * 22
else:
    numParameters = 3 * 52
# posePerturbed = torch.tensor(pose.cpu().numpy() + (np.random.rand(pose_size) - 0.5) * noiseLevel, dtype=torch.float64, device=device, requires_grad=True)
# Keep hand fixed
if cfg.bodyJointOnly:
    poseHands = pose[numParameters:].clone().detach()
    poseParams = torch.tensor(pose[:numParameters].cpu().numpy() + (np.random.rand(numParameters) - 0.5) * cfg.noiseLevel, dtype=torch.float64, device=device, requires_grad=True)
    posePerturbed = torch.cat([poseParams, poseHands])
else:
    poseParams  = torch.tensor(pose.cpu().numpy() + (np.random.rand(pose_size) - 0.5) * cfg.noiseLevel, dtype=torch.float64, device=device, requires_grad=True)
    posePerturbed = poseParams


# In[14]:


vertsPerturbed = smplsh(betas, posePerturbed, trans).type(torch.float32)
smplshMeshPerturbed = Meshes([vertsPerturbed], [smplsh.faces.to(device)], textures=textures)
fig, axs = plt.subplots(1, cfg.numCams)
fig.set_size_inches(cfg.numCams*2, 2)
loss = 0
with torch.no_grad():
    for iCam in range(cfg.numCams):
        R=camRTs[iCam]['R']
        T=camRTs[iCam]['T']
        image = renderer(meshes_world=smplshMeshPerturbed, R=R, T=T)
        axs[iCam].imshow(image[0,...,3].cpu().numpy())
        axs[iCam].axis('off')
        loss += torch.sum((refImgs[iCam][..., 3] - image[..., 3]) ** 2).item() / cfg.numCams
outInitalImgFile = join(outFolderForExperiment, 'ZInitalImg.png')
fig.savefig(outInitalImgFile, dpi=cfg.imgSize, transparent=True, bbox_inches='tight', pad_inches=0)
outInitalImgFilePdf = join(outFolderForExperiment, 'ZInitalImg.pdf')
fig.savefig(outInitalImgFilePdf, dpi=cfg.imgSize, transparent=True, bbox_inches='tight', pad_inches=0)


# In[15]:


# with torch.no_grad():
#     loss = torch.sum((imageRef[..., 3] - image[..., 3]) ** 2)
print('Inital loss:', loss)
poses = []
losses = []


# In[16]:


optimizer = torch.optim.Adam([poseParams], lr=cfg.learningRate)


# In[18]:


# loop = tqdm_notebook(range(cfg.numIterations))
loop = tqdm(range(cfg.numIterations))
for i in loop:
    optimizer.zero_grad()
    torch.cuda.empty_cache()

    if cfg.bodyJointOnly:
#         poseHands = pose[numBodyParameters:].clone().detach()
        posePerturbed = torch.cat([poseParams, poseHands])
    else:
        posePerturbed = poseParams
    lossVal = 0
    for iCam in range(cfg.numCams):
        vertsPerturbed = smplsh(betas, posePerturbed, trans).type(torch.float32)
        smplshMeshPerturbed = Meshes([vertsPerturbed], [smplsh.faces.to(device)], textures=textures)
        R=camRTs[iCam]['R']
        T=camRTs[iCam]['T']
        images = renderer(smplshMeshPerturbed,  R=R, T=T)

        loss = torch.sum((refImgs[iCam][..., 3] - images[..., 3]) ** 2) / cfg.numCams
        loss.backward()
        lossVal += loss.item()
    
    # targetImg = images[0, ..., :3]
    
    # loss, _ = model()
    
    # recordData
    losses.append(lossVal)
    poses.append(posePerturbed.cpu().detach().numpy())
    
#     for cam in cameras:
#         image = render(...)
#     loss.backward()
        
    optimizer.step()
    memStats = torch.cuda.memory_stats(device=device)
    memAllocated = memStats['active_bytes.all.current'] / 1000000
    loop.set_description('loss %.2f, poseDiff: %.2f, MemUsed:%.2fMB' % (lossVal, torch.sum((pose-posePerturbed)**2).item(), memAllocated))
    # descStr = 'loss %.2f, poseDiff: %.2f, MemUsed:%.2f' % (lossVal, torch.sum((pose-posePerturbed)**2).item(), memAllocated)
    if loss.item() < cfg.terminateLoss:
        break
    
    # Save outputs to create a GIF. 
    if i % cfg.plotStep == 0:
        # R = look_at_rotation(model.camera_position[None, :], device=model.device)
        # T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]   # (1, 3)
        # image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        # image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        # image = img_as_ubyte(image)
        # writer.append_data(image)
        torch.cuda.empty_cache()
        plt.close('all')

        fig, axs = plt.subplots(1, cfg.numCams)
        fig.set_size_inches(cfg.numCams*2, 2)
        with torch.no_grad():
            for iCam in range(cfg.numCams):
                R=camRTs[iCam]['R']
                T=camRTs[iCam]['T']
                image = renderer(meshes_world=smplshMeshPerturbed, R=R, T=T)
                axs[iCam].imshow(image[0,...,3].cpu().numpy())
                axs[iCam].axis('off')
                
        outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
        plt.savefig(outImgFile, dpi=cfg.imgSize, transparent=True, bbox_inches='tight', pad_inches=0)


# In[ ]:





# In[ ]:





# In[ ]:




