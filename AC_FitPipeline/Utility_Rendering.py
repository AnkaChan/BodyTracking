# add path for demo utils functions 
import sys
import os
import glob
import json
import numpy as np
from datetime import datetime
import cv2
import torch
from os.path import join
from pathlib import Path

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    SfMPerspectiveCameras,
)
# from iglhelpers import *

from matplotlib import pyplot as plt

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

vertex_ids = {
    'smplh': {
        'nose':		    332,
        'reye':		    6260,
        'leye':		    2800,
        'rear':		    4071,
        'lear':		    583,
        'rthumb':		6191,
        'rindex':		5782,
        'rmiddle':		5905,
        'rring':		6016,
        'rpinky':		6133,
        'lthumb':		2746,
        'lindex':		2319,
        'lmiddle':		2445,
        'lring':		2556,
        'lpinky':		2673,
        'LBigToe':		3216,
        'LSmallToe':	3226,
        'LHeel':		3387,
        'RBigToe':		6617,
        'RSmallToe':    6624,
        'RHeel':		6787
    },
    'smplx': {
        'nose':		    9120,
        'reye':		    9929,
        'leye':		    9448,
        'rear':		    616,
        'lear':		    6,
        'rthumb':		8079,
        'rindex':		7669,
        'rmiddle':		7794,
        'rring':		7905,
        'rpinky':		8022,
        'lthumb':		5361,
        'lindex':		4933,
        'lmiddle':		5058,
        'lring':		5169,
        'lpinky':		5286,
        'LBigToe':		5770,
        'LSmallToe':    5780,
        'LHeel':		8846,
        'RBigToe':		8463,
        'RSmallToe': 	8474,
        'RHeel':  		8635
    },
    'smplsh' : {
        "nose": 332,
        "reye": 6189,
        "leye": 2800,
        "rear": 4000,
        "lear": 583,
        "rthumb": 6120,
        "rindex": 5711,
        "rmiddle": 5834,
        "rring": 5945,
        "rpinky": 6062,
        "lthumb": 2746,
        "lindex": 2319,
        "lmiddle": 2445,
        "lring": 2556,
        "lpinky": 2673,
        "LBigToe": 3212,
        "LSmallToe": 3222,
        "LHeel": 3316,
        "RBigToe": 6747,
        "RSmallToe": 6737,
        "RHeel": 6622
        }
}

def loadCompressedFittingParam(file, readPersonalShape=False):
    fitParam = np.load(file)
    transInit = fitParam['trans']
    poseInit = fitParam['pose']
    betaInit = fitParam['beta']

    if readPersonalShape:
        personalShape = fitParam['personalShape']
        return transInit, poseInit, betaInit, personalShape
    else:
        return transInit, poseInit, betaInit

def renderImages(cams, renderer, mesh, cams_torch=None, cfg=None):
    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            meshTransformed = updataMeshes(mesh, cams_torch, iCam, cfg)

            image_cur = renderer.renderer(meshTransformed, cameras=cams[iCam])
            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)
    # showCudaMemUsage(device)
    return images

def updataMeshes(meshes, cams_torch, iCam, cfg):
    # apply extrinsics outside camera to avoid bugs
    if cfg.extrinsicsOutsideCamera:
        R = cams_torch['R'][iCam*cfg.batchSize:iCam*cfg.batchSize + cfg.batchSize].to(meshes.device)
        T = cams_torch['T'][iCam*cfg.batchSize:iCam*cfg.batchSize + cfg.batchSize].to(meshes.device)

        transposed = torch.transpose(meshes.verts_padded(), 1, 2)
        R = torch.transpose(R, 1, 2)

        vertsTransformed = torch.matmul(R, transposed) + T[..., None]
        vertsTransformed = torch.transpose(vertsTransformed, 1, 2)
        meshes = meshes.update_padded(vertsTransformed)
    return meshes

def renderImagesWithBackground(cams, renderer, mesh, backgrounds, device=None, cams_torch=None, cfg=None):
    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            meshTransformed = updataMeshes(mesh, cams_torch, iCam, cfg)

            if device is not None:
                showCudaMemUsage(device)
            blend_params = BlendParams(
                renderer.blend_params.sigma, renderer.blend_params.gamma, background_color=backgrounds[iCam])
            image_cur = renderer.renderer(meshTransformed, cameras=cams[iCam], blend_params=blend_params)

            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)
        # showCudaMemUsage(device)
    return images


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

def visualize2DResults(images, backGroundImages=None, outImgFile=None, rows=2, sizeInInches=2, withAlpha=True):
    lossVal = 0
    numCams = len(images)
    numCols = int(numCams / rows)
    fig, axs = plt.subplots(rows, numCols)
    fig.set_size_inches(numCols * sizeInInches, rows * sizeInInches)
    with torch.no_grad():
        for iRow in range(rows):
            for iCol in range(numCols):
                iCam = rows * iRow + iCol
                imgAlpha = images[iCam]

                if backGroundImages is not None:
                    img = np.copy(backGroundImages[iCam])
                    #                     fgMask = np.logical_not(np.where())
                    for iChannel in range(3):
                        img[..., iChannel] = np.where(imgAlpha, imgAlpha, backGroundImages[iCam][..., iChannel])
                    imgAlpha = img

                imgAlpha = cv2.flip(imgAlpha, -1)
                if not withAlpha:
                    imgAlpha = imgAlpha[...,:3]

                axs[iRow, iCol].imshow(imgAlpha, vmin=0.0, vmax=1.0)
                axs[iRow, iCol].axis('off')

        if outImgFile is not None:
            fig.savefig(outImgFile, dpi=512, transparent=True, bbox_inches='tight', pad_inches=0)

def visualize2DSilhouetteResults(images, backGroundImages=None, outImgFile=None, rows=2,
                                 sizeInInches=2):
    numCams = len(images)
    numCols = int(numCams / rows)
    fig, axs = plt.subplots(rows, numCols)
    fig.set_size_inches(numCols * sizeInInches, rows * sizeInInches)
    with torch.no_grad():
        for iRow in range(rows):
            for iCol in range(numCols):
                iCam = rows * iRow + iCol
                imgAlpha = images[iCam, ..., 3]

                if backGroundImages is not None:
                    img = np.copy(backGroundImages[iCam]) * 0.5
                    #                     fgMask = np.logical_not(np.where())
                    #                     for iChannel in range(3):
                    img[..., 0] = img[..., 0] + imgAlpha * 0.5
                    imgAlpha = img

                imgAlpha = cv2.flip(imgAlpha, -1)

                axs[iRow, iCol].imshow(imgAlpha, vmin=0.0, vmax=1.0)
                axs[iRow, iCol].axis('off')

        if outImgFile is not None:
            fig.savefig(outImgFile, dpi=512, transparent=True, bbox_inches='tight', pad_inches=0)

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
	
def load_cameras(cam_path, device, actual_img_shape, unitM=False):
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
        if unitM:
            T = T/1000

        R, _ = cv2.Rodrigues(rvec)

        Rs.append((R).T)
        Ts.append(T)

        cx_corrected = (cx*2/img_size - w/img_size)
        cy_corrected = (cy*2/img_size - h/img_size)
        fx_corrected = (fx*2/img_size)
        fy_corrected = (fy*2/img_size)

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

def load_image(img_dir, cropSize=1080, flipImg=False, normalize = False, cvtToRGB=False):
    img = cv2.imread(img_dir)
    if normalize:
        img = img.astype(np.float32) / 255.0
    if cvtToRGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w = int(img.shape[0]) / 2


    image = img
    cx = image.shape[1] / 2

    image = image[:, int(cx - w):int(cx + w)]
    if not cropSize == img.shape[0]:
        crop_out = cv2.resize(image, (cropSize, cropSize))
    if flipImg:
        crop_out = cv2.flip(crop_out, -1)

    return img, crop_out


def load_images(img_dir, UndistImgs=False, camParamF=None, cropSize=2160, imgExt='png', writeUndistorted=True,
                normalize=True, flipImg=True, cvtToRGB=True):
    image_refs_out = []
    crops_out = []
    undistImageFolder = join(img_dir, 'Undist')

    if UndistImgs:
        os.makedirs(undistImageFolder, exist_ok=True)
        camParams = json.load(open(camParamF))['cam_params']

    # for img_name in img_names:
    #    path = img_dir + '\\{}'.format(img_name)
    #    print(path)
    img_paths = sorted(glob.glob(img_dir + '/*.' + imgExt))

    for i, path in enumerate(img_paths):
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.imread(path)

        if UndistImgs:
            # f = inFiles[iCam][iP]
            fx = camParams[str(i)]['fx']
            fy = camParams[str(i)]['fy']
            cx = camParams[str(i)]['cx']
            cy = camParams[str(i)]['cy']
            intrinsic_mtx = np.array([
                [fx, 0.0, cx, ],
                [0.0, fy, cy],
                [0.0, 0.0, 1],
            ])

            undistortParameter = np.array(
                [camParams[str(i)]['k1'], camParams[str(i)]['k2'], camParams[str(i)]['p1'], camParams[str(i)]['p2'],
                 camParams[str(i)]['k3'], camParams[str(i)]['k4'], camParams[str(i)]['k5'], camParams[str(i)]['k6']])

            img = cv2.undistort(img, intrinsic_mtx, undistortParameter)
            if writeUndistorted:
                outUndistImgFile = join(undistImageFolder, Path(path).stem + '.png')
                cv2.imwrite(outUndistImgFile, img)
        if normalize:
            img = img.astype(np.float32) / 255.0
        if cvtToRGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_refs_out.append(img)

    w = int(img.shape[0]) / 2

    for i in range(len(image_refs_out)):
        image = image_refs_out[i]
        cx = image.shape[1] / 2

        image = image_refs_out[i]
        img = image[:, int(cx - w):int(cx + w)]
        if not cropSize == img.shape[0]:
            img = cv2.resize(img, (cropSize, cropSize))
        if flipImg:
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

def init_camera_batches(cam_torch, device, batchSize = 1, withoutExtrinsics=False):
    cams = []
    numCams = cam_torch['R'].shape[0]

    numBatches = int(numCams / batchSize)
    for i in range(numBatches):
        focal_length = cam_torch['fl'][i*batchSize:i*batchSize+batchSize]
        principal_point = cam_torch['pp'][i*batchSize:i*batchSize+batchSize]
        if withoutExtrinsics:
            R = torch.tensor(np.repeat(np.eye(3)[None, ...], batchSize, axis=0), device=device)
            T = torch.tensor(np.repeat(np.zeros(3)[None, ...], batchSize, axis=0), device=device)
        else:
            R = cam_torch['R'][i*batchSize:i*batchSize+batchSize]
            T = cam_torch['T'][i*batchSize:i*batchSize+batchSize]
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


def getLaplacian(meshFile, biLaplacian = False):
    import pyigl as igl

    extName = Path(meshFile).suffix
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    if extName.lower() == '.obj':
        igl.readOBJ(meshFile, V, F)
    elif extName.lower()  == '.ply':
        N = igl.eigen.MatrixXd()
        UV = igl.eigen.MatrixXd()
        igl.readPLY(meshFile, V, F, N, UV)

    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()

    igl.cotmatrix(V, F, L)

    LNP = - e2p(L).todense()
    if biLaplacian:
        LNP = LNP @ LNP

    return LNP

def to_tensor(array, device='cpu', dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype, device=device)
    else:
        return array

def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

class VertexJointSelector(torch.nn.Module):

    def __init__(self, vertex_ids=None,
                 use_hands=True,
                 use_feet_keypoints=True, **kwargs):
        super(VertexJointSelector, self).__init__()

        extra_joints_idxs = []

        face_keyp_idxs = np.array([
            vertex_ids['nose'],
            vertex_ids['reye'],
            vertex_ids['leye'],
            vertex_ids['rear'],
            vertex_ids['lear']], dtype=np.int64)

        extra_joints_idxs = np.concatenate([extra_joints_idxs,
                                            face_keyp_idxs])

        if use_feet_keypoints:
            feet_keyp_idxs = np.array([vertex_ids['LBigToe'],
                                       vertex_ids['LSmallToe'],
                                       vertex_ids['LHeel'],
                                       vertex_ids['RBigToe'],
                                       vertex_ids['RSmallToe'],
                                       vertex_ids['RHeel']], dtype=np.int32)

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, feet_keyp_idxs])

        if use_hands:
            self.tip_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

            tips_idxs = []
            for hand_id in ['l', 'r']:
                for tip_name in self.tip_names:
                    tips_idxs.append(vertex_ids[hand_id + tip_name])

            extra_joints_idxs = np.concatenate(
                [extra_joints_idxs, tips_idxs])

        self.register_buffer('extra_joints_idxs',
                             to_tensor(extra_joints_idxs,  dtype=torch.long))

    def forward(self, vertices, joints):
        extra_joints = torch.index_select(vertices, 1, self.extra_joints_idxs.to(vertices.device), )
        joints = torch.cat([joints, extra_joints], dim=1)

        return joints

class JointMapper(torch.nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps.to(joints.device))

class VertexToOpJointsConverter(torch.nn.Module):
    def __init__(s, **kwargs):
        super(VertexToOpJointsConverter, s).__init__(**kwargs)

        s.jSelector = VertexJointSelector(vertex_ids['smplsh'])
        jointMap = smpl_to_openpose('smplh', use_hands=True,
                                    use_face=False,
                                    use_face_contour=False, )

        s.joint_mapper = JointMapper(jointMap)

    def forward(s, smplshVerts, smplshJoints):
        allJoints = s.jSelector(to_tensor(smplshVerts, smplshVerts.device), smplshJoints)
        joint_mapped = s.joint_mapper(allJoints)

        return joint_mapped

def write_obj(file_name, verts, faces=None, faceVIdAdd1 = True):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        if faces is not None:
            for f in faces:
                fp.write('f ')
                for vId in f:
                    if faceVIdAdd1:
                        vId += 1
                    fp.write('%d ' % (vId))
                fp.write('\n')

