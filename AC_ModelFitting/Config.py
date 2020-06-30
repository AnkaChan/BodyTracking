SMPLSH_Dir = r'..\SMPL_reimp'

import sys
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch

import os
import torch
import matplotlib.pyplot as plt
from skimage.io import imread
import torch.nn as nn
import numpy as np
from skimage import img_as_ubyte
import imageio
import json
import cv2
import time
from PIL import Image
from pytorch3d.loss import (
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
import Utility
import torch.nn.functional as F

from tqdm import tqdm_notebook
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply
import math
# Data structures and functions for rendering
from pytorch3d.structures import Meshes, Textures, join_meshes_as_batch
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
    SfMOrthographicCameras,
    PointLights,
    BlendParams,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturedSoftPhongShader,
    SoftSilhouetteShader,
    look_at_rotation,
    HardFlatShader
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

import pyvista as pv

import Logger

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
        s.terminateLoss = 0.1
        s.plotStep = 10
        s.numCams = 16
        s.imgSize = 2160

        s.lpSmootherW = 0.1
        s.normalSmootherW = 0.1

        s.biLaplacian = False
        s.jointRegularizerWeight = 0.000001

        # fix to keypoint
        s.useKeypoints = False
        s.kpFixingWeight = 1
        # fix the shape of hand and head
        s.vertexFixingWeight = 100
        s.toSparseCornersFixingWeight = 1e-6

        # for per vertex adjustment only
        s.optimizePose = False
        s.bin_size=0
        s.cull_backfaces=False

class Renderer:
    def __init__(s, device, cfg=RenderingCfg()):
        s.cfg = cfg
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        s.blend_params = BlendParams(sigma=cfg.sigma, gamma=1e-4)

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        s.lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
        # cameras = OpenGLPerspectiveCameras(device=device)
        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model

        if cfg.blurRange != 0:
            s.raster_settings = RasterizationSettings(
                image_size=cfg.imgSize,
                blur_radius=np.log(1. / cfg.blurRange - 1.) * s.blend_params.sigma,
                faces_per_pixel=cfg.faces_per_pixel,
                bin_size=cfg.bin_size,
                cull_backfaces=cfg.cull_backfaces
            )
        else:
            s.raster_settings = RasterizationSettings(
                image_size=cfg.imgSize,
                blur_radius=0,
                faces_per_pixel=cfg.faces_per_pixel,
                bin_size=cfg.bin_size,
                cull_backfaces=cfg.cull_backfaces

            )

        s.rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=s.raster_settings
        )
        if cfg.blurRange != 0:
            s.renderer = MeshRenderer(
                rasterizer=s.rasterizer,
                #     shader=SoftPhongShader(
                #         device=device,
                #         cameras=cameras,
                #         lights=lights,
                #         blend_params=blend_params
                #     )
                shader=SoftSilhouetteShader(
                    blend_params=s.blend_params
                    # device=device,
                    # cameras=cameras,
                    # lights=lights
                )
            )
        else:
            s.renderer = MeshRenderer(
                rasterizer=s.rasterizer,
                #     shader=SoftPhongShader(
                #         device=device,
                #         cameras=cameras,
                #         lights=lights,
                #         blend_params=blend_params
                #     )
                shader=SoftSilhouetteShader(
                    blend_params=s.blend_params
                    # device=device,
                    # cameras=cameras,
                    # lights=lights
                )
            )

class RendererWithTexture:
    def __init__(s, device, lights= None, cfg=RenderingCfg()):
        s.cfg = cfg
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        s.blend_params = BlendParams(sigma=cfg.sigma, gamma=1e-4)

        # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
        # -z direction.
        if lights is None:
            s.lights = PointLights(device=device, location=[[0.0, 0.0, -3000.0]])
        else:
            s.lights = lights
        # cameras = OpenGLPerspectiveCameras(device=device)
        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model

        if cfg.blurRange != 0:
            s.raster_settings = RasterizationSettings(
                image_size=cfg.imgSize,
                blur_radius=np.log(1. / cfg.blurRange - 1.) * s.blend_params.sigma,
                faces_per_pixel=cfg.faces_per_pixel,

                bin_size=cfg.bin_size,
                cull_backfaces=cfg.cull_backfaces

            )
        else:
            s.raster_settings = RasterizationSettings(
                image_size=cfg.imgSize,
                blur_radius=0,
                faces_per_pixel=cfg.faces_per_pixel,
                bin_size=cfg.bin_size,
                cull_backfaces=cfg.cull_backfaces

            )

        s.rasterizer = MeshRasterizer(
            cameras=None,
            raster_settings=s.raster_settings
        )
        if cfg.blurRange != 0:
            s.renderer = MeshRenderer(
                rasterizer=s.rasterizer,
                #     shader=SoftPhongShader(
                #         device=device,
                #         cameras=cameras,
                #         lights=lights,
                #         blend_params=blend_params
                #     )
                # shader=SoftSilhouetteShader(
                #     blend_params=s.blend_params
                #     # device=device,
                #     # cameras=cameras,
                #     # lights=lights
                # )
                shader=TexturedSoftPhongShader(
                    blend_params=s.blend_params,
                    device=device,
                    lights=s.lights
                )
            )
        else:
            s.renderer = MeshRenderer(
                rasterizer=s.rasterizer,
                #     shader=SoftPhongShader(
                #         device=device,
                #         cameras=cameras,
                #         lights=lights,
                #         blend_params=blend_params
                #     )
                shader=TexturedSoftPhongShader(
                    blend_params=s.blend_params,
                    device=device,
                    lights=s.lights
                )
                # shader=SoftSilhouetteShader(
                #     blend_params=s.blend_params
                #     # device=device,
                #     # cameras=cameras,
                #     # lights=lights
                # )
            )