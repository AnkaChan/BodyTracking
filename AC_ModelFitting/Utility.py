# add path for demo utils functions 
import sys
import os
import glob
import json
import numpy as np
from datetime import datetime
import cv2
import torch

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
)

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

def reproject(params, vertices, distort=False):
    R = params['R']
    T = params['T']
    fx = params['fx']
    fy = params['fy']
    cx = params['cx']
    cy = params['cy']

    E = np.array([
        [R[0,0], R[0,1], R[0,2], T[0]], 
        [R[1,0], R[1,1], R[1,2], T[1]], 
        [R[2,0], R[2,1], R[2,2], T[2]], 
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
            r2 = xp**2 + yp**2
            ## radial
            radial_dist = 1 + k1*(r2) + k2*(r2*r2) + k3*(r2*r2*r2)

            ## tangential
            tan_x = p2 * (r2 + 2.0 * xp * xp) + 2.0 * p1 * xp * yp
            tan_y = p1 * (r2 + 2.0 * yp * yp) + 2.0 * p2 * xp * yp

            xp = xp * radial_dist + tan_x
            yp = yp * radial_dist + tan_y
            
        u = fx * xp + cx
        v = fy * yp + cy
        pr = 1
        nr = 0
        if (-4000*nr < u and u < pr*4000) and (-2160*nr < v and v < pr*2160):
            img_pts.append(np.array([u, v]))
    img_pts = np.array(img_pts)
    return img_pts
	
def load_cameras(cam_path, device, actual_img_shape):
    print('actual_img_shape:',actual_img_shape)
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
        
        cx_corrected = cx*2/img_size - w/img_size
        cy_corrected = cy*2/img_size - h/img_size
        fx_corrected = fx*2/img_size
        fy_corrected = fy*2/img_size
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

def load_images(img_dir, UndistImgs = False, cropSize = 2160, imgExt = 'png'):
    image_refs_out = []
    crops_out = []
    
    #for img_name in img_names:
    #    path = img_dir + '\\{}'.format(img_name)
    #    print(path)
    img_paths = sorted(glob.glob(img_dir + '/*.' + imgExt))
    
    for i, path in enumerate(img_paths):
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.imread(path).astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_refs_out.append(img)

    w = int(img.shape[0])/2

    for i in range(len(image_refs_out)):
        image = image_refs_out[i]
        cx = image.shape[1] / 2
        
        image = image_refs_out[i]
        img = image[:, int(cx-w):int(cx+w)]
        if not cropSize == w:
            img = cv2.resize(img, (cropSize, cropSize))
        img = cv2.flip(img, -1)
        crops_out.append(img)

    return image_refs_out, crops_out

def normalizeNormals(normals):
    norm = torch.sqrt(normals[:, 0]**2 + normals[:, 1]**2 + normals[:, 2]**2)
    print(norm.shape)
    normals[:, 0] = normals[:, 0] / norm
    normals[:, 1] = normals[:, 1] / norm
    normals[:, 2] = normals[:, 2] / norm
    return normals

def init_camera_batches(cam_torch, device):
    cams = []
    numCams = cam_torch['R'].shape[0]
    for i in range(numCams):
        focal_length = cam_torch['fl'][i:i+1]
        principal_point = cam_torch['pp'][i:i+1]
        R = cam_torch['R'][i:i+1]
        T = cam_torch['T'][i:i+1]
        cameras = SfMPerspectiveCameras(device=device, R=R, T=T, principal_point=principal_point, focal_length=focal_length)
        cams.append(cameras)
    return cams

def saveVTK(outFile, verts, smplshExampleMesh):
    smplshExampleMesh.points = verts
    smplshExampleMesh.save(outFile)

def showCudaMemUsage(device):
    memStats = torch.cuda.memory_stats(device=device)
    print('Before release: active_bytes.all.current:', memStats['active_bytes.all.current'] / 1000000, 'MB')
    torch.cuda.empty_cache()
    memStats = torch.cuda.memory_stats(device=device)
    print('After release: active_bytes.all.current:', memStats['active_bytes.all.current'] / 1000000, 'MB')