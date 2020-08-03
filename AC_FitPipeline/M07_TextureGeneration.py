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

import torch.nn.functional as F

from tqdm import tqdm_notebook
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
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

# add path for demo utils functions
import sys
import os
import glob

sys.path.append(os.path.abspath(''))

print(torch.version.cuda)
from datetime import datetime


def now_str():
    now = datetime.now()
    month = str(now.month)
    day = str(now.day)
    hour = str(now.hour)
    minute = str(now.minute)
    sec = str(now.second)

    output = '[{:>02}/{:>02} {:>02}:{:>02}:{:>02}]'.format(month, day, hour, minute, sec)
    return output


def __output_log(path, strs):
    if not os.path.exists(path):
        with open(path, 'w+') as f:
            f.write(strs)
            f.close()
    else:
        with open(path, 'a+') as f:
            f.write(strs)
            f.close()


print(now_str())
print(torch.__version__)


def reproject(params, vertices, distort=False):
    R = params['R']
    T = params['T']
    fx = params['fx']
    fy = params['fy']
    cx = params['cx']
    cy = params['cy']

    E = np.array([
        [R[0, 0], R[0, 1], R[0, 2], T[0]],
        [R[1, 0], R[1, 1], R[1, 2], T[1]],
        [R[2, 0], R[2, 1], R[2, 2], T[2]],
        [0, 0, 0, 1]]).astype('double')

    if distort:
        k1 = params['k1']
        k2 = params['k2']
        k3 = params['k3']
        p1 = params['p1']
        p2 = params['p2']

    img_pts = []
    for i in range(len(vertices)):
        v = np.array(vertices[i])

        # extrinsics
        v4 = E.dot(np.array([v[0], v[1], v[2], 1]).astype('double'))
        xp = v4[0] / v4[2]
        yp = v4[1] / v4[2]

        if distort:
            # intrinsics
            r2 = xp ** 2 + yp ** 2
            ## radial
            radial_dist = 1 + k1 * (r2) + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)

            ## tangential
            tan_x = p2 * (r2 + 2.0 * xp * xp) + 2.0 * p1 * xp * yp
            tan_y = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp

            xp = xp * radial_dist + tan_x
            yp = yp * radial_dist + tan_y

        u = fx * xp + cx
        v = fy * yp + cy
        pr = 1
        nr = 0
        if (-4000 * nr < u and u < pr * 4000) and (-2160 * nr < v and v < pr * 2160):
            img_pts.append(np.array([u, v]))
    img_pts = np.array(img_pts)
    return img_pts


def load_cameras(cam_path, device, actual_img_shape):
    print('actual_img_shape:', actual_img_shape)
    h = actual_img_shape[0]
    w = actual_img_shape[1]
    img_size = min(w, h)

    # load cameras
    cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    with open(cam_path, 'r') as f:
        j = json.load(f)
        camera_params = j['cam_params']

    cam_params = []
    Rs, Ts, focal_lengths, principal_points = [], [], [], []
    for cam_idx, cam in enumerate(cams):
        cam_param = camera_params[str(cam_idx)]
        # for undistortion
        fx = cam_param['fx']
        fy = cam_param['fy']
        cx = cam_param['cx']
        cy = cam_param['cy']
        k1 = cam_param['k1']
        k2 = cam_param['k2']
        k3 = cam_param['k3']
        p1 = cam_param['p1']
        p2 = cam_param['p2']

        rvec = np.float32(cam_param['rvec'])
        T = np.float32(cam_param['tvec'])
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R.T)
        Ts.append(T)

        cx_corrected = cx * 2 / img_size - w / img_size
        cy_corrected = cy * 2 / img_size - h / img_size
        fx_corrected = fx * 2 / img_size
        fy_corrected = fy * 2 / img_size
        principal_point = np.array([cx_corrected, cy_corrected]).astype(np.float32)
        focal_length = np.array([fx_corrected, fy_corrected]).astype(np.float32)
        focal_lengths.append(focal_length)
        principal_points.append(principal_point)

        K = np.float32([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.float32([k1, k2, p1, p2, k3])
        cam_params.append({'K': K, 'dist': dist, 'R': R, 'T': T, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})

    R_torch = torch.from_numpy(np.array(Rs).astype(np.float32))
    T_torch = torch.from_numpy(np.array(Ts).astype(np.float32))
    focal_length = torch.from_numpy(np.array(focal_lengths).astype(np.float32))
    principal_point = torch.from_numpy(np.array(principal_points).astype(np.float32))
    out_for_torch = {'R': R_torch, 'T': T_torch, 'fl': focal_length, 'pp': principal_point}
    return cam_params, out_for_torch



class Config:
    def __init__(s):
        s.texturemap_shape = (1024, 1024, 3)
        s.image_size = 1080

        # input image size
        s.actual_img_shape = (2160, 4000)
        s.batch_size = 1
        s.erosionSize = 3

def gen_target_images(device, render_data, image_size):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0,
        faces_per_pixel=1,
        bin_size=0,  # this setting controls whether naive or coarse-to-fine rasterization is used
        max_faces_per_bin=None  # this setting is for coarse rasterization
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(BlendParams(sigma=1e-4, gamma=1e-4))
    )

    images = []
    for d in render_data:
        meshes = d['meshes']
        cameras = d['cameras']
        image = renderer(meshes_world=meshes, cameras=cameras)[..., 3]
        images.append(image.detach().squeeze().cpu().numpy())
    images = np.stack(images, axis=0).astype(np.float32)
    return images


def init_camera_batches(device, cams_torch, n_batch, batch_size):
    cams = []
    for batch_idx in range(n_batch):
        i0 = batch_idx * batch_size
        i1 = i0 + batch_size
        R = cams_torch['R'][i0:i1]
        T = cams_torch['T'][i0:i1]
        focal_length = cams_torch['fl'][i0:i1]
        principal_point = cams_torch['pp'][i0:i1]

        cameras = SfMPerspectiveCameras(device=device, R=R, T=T, principal_point=principal_point,
                                        focal_length=focal_length)
        cams.append(cameras)
    return cams


def load_mesh(device, mesh_path):
    verts, faces, aux = load_obj(mesh_path)
    faces_idx = faces.verts_idx

    verts_uvs = aux.verts_uvs[None, ...].to(device)  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...].to(device)  # (1, F, 3)

    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_idx.to(device)]
    )
    return mesh


def binarize(np_imgs):
    out = []
    for i in range(np_imgs.shape[0]):
        out.append((np_imgs[i] > 0).astype(np.float32))
    return np.float32(out)

def generateContourMask(cam_path, mesh_dir, frameNames, clean_plate_dir, device, out_dir, cfg=Config()):
    mesh_paths = []
    # cams = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    for img_name in frameNames:
        mesh_paths.append(mesh_dir + '\\{}.obj'.format(img_name))
        print(mesh_paths[-1])
    # n_forwards = len(frameNames) * len(cams)

    cam_params, cams_torch = load_cameras(cam_path, device, cfg.actual_img_shape)
    print(len(cam_params), ':', cam_params[0].keys())
    print(cams_torch.keys())
    print(cams_torch['R'].shape)

    n_forwards = len(frameNames) * len(cam_params)

    n_batch = int(n_forwards / cfg.batch_size)
    print('{} renderes'.format(n_forwards), ', n_batch={}, batch_size={}'.format(n_batch, cfg.batch_size))

    image_refs = {}
    for img_idx, img_name in enumerate(frameNames):
        print(img_name)
        mesh_path = mesh_paths[img_idx]
        mesh_target = load_mesh(device, mesh_path)

        n_batch_temp = 16
        batch_size_temp = 1
        cameras = init_camera_batches(device, cams_torch, n_batch_temp, batch_size_temp)

        render_data = []
        for i in range(len(cameras)):
            data = {'meshes': mesh_target.extend(cfg.batch_size), 'cameras': cameras[i]}
            render_data.append(data)

        img_ref = gen_target_images(device, render_data, cfg.image_size)
        img_ref = binarize(img_ref)
        image_refs[img_name] = img_ref
    # for k, v in image_refs.items():
    #     print(k, v.shape)

    # get contours
    kernel = np.ones((cfg.erosionSize, cfg.erosionSize), np.uint8)
    contours = {}
    for img_name in frameNames:
        cnts = []
        for i in range(image_refs[img_name].shape[0]):
            img = (image_refs[img_name][i]*255).astype(np.uint8)
            dilation = cv2.dilate(img, kernel, iterations=1)
            erosion = cv2.erode(img, kernel, iterations=1)
            #cnt = img - erosion
            cnt = dilation - erosion
            cnts.append(cnt)
        contours[img_name] = np.float32(cnts)
    # for k, v in contours.items():
    #     print(k, v.shape)
    # n_imgs = contours[img_name].shape

    os.makedirs(out_dir, exist_ok=True)
    for k, cnts in contours.items():
        # print(k)
        for img_idx in range(cnts.shape[0]):
            out_path = out_dir + '/{}_{}.png'.format(k, img_idx)
            img = cv2.flip(cnts[img_idx], -1)
            cv2.imwrite(out_path, img)
        print(out_path)
    print('Done')