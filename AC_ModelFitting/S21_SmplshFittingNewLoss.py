import pyvista as pv
from scipy.spatial.transform import Rotation as R
SMPLSH_Dir = r'..\SMPL_reimp'
import sys, glob
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch
import numpy as np
from iglhelpers import *
import pyigl as igl
import tqdm, os, json
from os.path import join
import vtk
from pathlib import Path
import pickle
from S19_SMPLHJToOPJointRegressor import *
import Utility
from SuiteCapture import Camera
import cv2

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

def saveObj(path, verts, faces, faceIdAdd1 = True):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            if faceIdAdd1:
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
            else:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

class KeypointsLoss2D(torch.nn.Module):
    def __init__(s, device, camParams, minConfidence=0.2, dtype=torch.float32, **kwargs):
        super(KeypointsLoss2D, s).__init__(**kwargs)
        s.device = device
        s.minConfidence = minConfidence
        s.dtype = dtype
        # from camParams to projection matrix
        s.projMats = []
        s.intrinsicMats = []
        s.undistParams = []
        for i in range(len(camParams)):
            I, E = Camera.calibrationParamsToIEMats(camParams[i], True)
            s.intrinsicMats.append(I[:3,:3])
            s.projMats.append(I @ E)
            camParam = camParams[i]
            undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
                                           camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']],
                                          dtype=np.float32)
            s.undistParams.append(undistortParameter)

        # s.intrinsicMats = torch.tensor(s.intrinsicMats, dtype=dtype, requires_grad=True, device=s.device)
        s.projMats = torch.tensor(s.projMats, dtype=dtype, requires_grad=False, device=s.device)

    def forward(s, kps3D, kp2DDetected, undist2DKp = False, normalize=True):
        weights = kp2DDetected[..., 2]
        weights[weights < s.minConfidence] = 0
        weights = torch.tensor(weights, dtype=s.dtype, requires_grad=False, device=s.device)

        kps2D = []
        if undist2DKp:
            for iCam in range(kp2DDetected.shape[0]):
                keypointsPadded = kp2DDetected[iCam, :, None, :2]
                keypointsPaddedUndist = cv2.undistortPoints(keypointsPadded, s.intrinsicMats[iCam],
                                    s.undistParams[iCam], P=s.intrinsicMats[iCam])
                keypointsPaddedUndist = np.squeeze(keypointsPaddedUndist)
                kps2D.append(keypointsPaddedUndist)
        else:
            for iCam in range(kp2DDetected.shape[0]):
                kps2D.append(kp2DDetected[iCam, :, :2])

        kps2D = torch.tensor(kps2D, dtype=s.dtype, requires_grad=False, device=s.device)

        # slower verson, for loop
        # for iCam in range(kp2DDetected.shape[0]):
        kps3DH = torch.cat([
                torch.transpose(kps3D, 0, 1), torch.ones((1, kps3D.shape[0]), dtype=s.dtype, requires_grad=False, device=s.device)
            ], dim=0)
        kpProj2D = torch.einsum('bki,ij->bkj', [s.projMats, kps3DH])
        kpProj2D = kpProj2D[:, :2, :] / kpProj2D[:, 2:3, :]
        kpProj2D = torch.transpose(kpProj2D, 1, 2)

        if normalize:
            actual_img_shape = (2160, 4000)
            kpProj2D[..., 0] = kpProj2D[..., 0] / actual_img_shape[1]
            kpProj2D[..., 1] = kpProj2D[..., 1] / actual_img_shape[0]
            kps2D[..., 0] = kps2D[..., 0] / actual_img_shape[1]
            kps2D[..., 1] = kps2D[..., 1] / actual_img_shape[0]

        distance = torch.mean(((kpProj2D - kps2D) * weights[...,None])** 2)
        return distance



if __name__ == '__main__':
    kpFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03052\toRGB\KeypointsJson'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # smplshExampleFile = r'..\Data\BuildSmplsh\Input\SMPLWithSocks_tri_Aligned.obj'
    smplshExampleFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\03052.obj'
    smplshDataFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    outFolder = r'F:\WorkingCopy2\2020_07_09_TestSimplify\SimplifyLossFitting\03052'

    # test new smplsh data
    inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # torch.cuda.set_device(device)
    learningRate = 1e-5
    numIterations = 100000

    kpFiles = glob.glob(join(kpFolder, '*.json'))
    kpFiles.sort()

    kp2DMultiCams = []
    for kpF in kpFiles:
        kpData = json.load(open(kpF))
        # print(kpData)
        body_keypoints_multiPerson = []
        for idx, person_data in enumerate(kpData['people']):
            body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                      dtype=np.float32)
            body_keypoints = body_keypoints.reshape([-1, 3])
            left_hand_keyp = np.array(
                person_data['hand_left_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                person_data['hand_right_keypoints_2d'],
                dtype=np.float32).reshape([-1, 3])
            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)

            body_keypoints_multiPerson.append(body_keypoints)

        # print(body_keypoints_multiPerson[0])
        kp2DMultiCams.append(body_keypoints_multiPerson[0])

    actual_img_shape = (2160, 4000)
    # camParams = Utility.load_cameras(camParamF, device, actual_img_shape)
    camParams=json.load(open(camParamF))
    camParams = [camParams['cam_params'][str(i)] for i in range(16)]

    kpLoss = KeypointsLoss2D(device, camParams, minConfidence=0.5)

    smplshMesh = pv.PolyData(smplshExampleFile)

    smplshVerts = to_tensor(smplshMesh.points, device)
    smplsh = smplsh_torch.SMPLModel(device, smplshDataFile, personalShape=None)

    joints = smplsh.J_regressor.type(torch.float32) @ smplshVerts

    jointConverter = Utility.VertexToOpJointsConverter()
    opJoints = jointConverter(smplshVerts[None,...], joints[None,...])

    kp2DMultiCams = np.array(kp2DMultiCams)
    kp2DMultiCams[:, :, :2] = kp2DMultiCams[:, :, :2] * 2

    loss = kpLoss(opJoints[0, ...], kp2DMultiCams, undist2DKp=True)
    # loss = kpLoss(opJoints[0, ...], kp2DMultiCams, undist2DKp=False)

    print('loss:', loss)

    transInit, poseInit, betaInit = loadCompressedFittingParam(inFittingParam, readPersonalShape=False)
    # Make fitting parameter tensors
    pose = torch.tensor(poseInit , dtype=torch.float64, requires_grad=True, device=device)
    # pose = torch.tensor(poseInit + 0.2 * np.random.randn(*poseInit.shape), dtype=torch.float64, requires_grad=True, device=device)
    # pose = torch.tensor(poseInit * 0, dtype=torch.float64, requires_grad=True, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=True, device=device)
    # betas = torch.tensor(betaInit + 0.2 * np.random.randn(*betaInit.shape), dtype=torch.float64, requires_grad=True, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=True, device=device)
    verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)

    optimizer = torch.optim.Adam([trans, pose, betas], lr=learningRate)

    loop = tqdm.tqdm(range(numIterations))

    os.makedirs(outFolder, exist_ok=True)
    # main optimization loop
    for i in loop:
        optimizer.zero_grad()
        verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)
        verts = verts.type(torch.float32)
        jointsDeformed = jointsDeformed.type(torch.float32)
        opJoints = jointConverter(verts[None, ...], jointsDeformed[None, ...])

        loss = kpLoss(opJoints[0, ...]*1000, kp2DMultiCams, undist2DKp=True)
        loop.set_description('Step: ' + str(i) + ' loss: ' + str(loss.item()))
        loss.backward()
        optimizer.step()

        if not (1+i) % 2000 and i:
            smplsh.write_obj(verts, join(outFolder, 'Opt_' + str(i).zfill(5) + '.obj'))