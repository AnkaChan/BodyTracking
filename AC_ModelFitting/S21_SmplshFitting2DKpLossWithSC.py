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

class GMoF(torch.nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        # return self.rho ** 2 * dist

        return dist

class KeypointsLoss2D(torch.nn.Module):
    def __init__(s, device, camParams, kp2DDetected, undist2DKp = False,  GMRho=10, minConfidence=0.2, dtype=torch.float32, **kwargs):
        super(KeypointsLoss2D, s).__init__(**kwargs)
        s.device = device
        s.minConfidence = minConfidence
        s.dtype = dtype
        s.kp2DDetected = kp2DDetected
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

        s.kps2D = []
        if undist2DKp:
            for iCam in range(kp2DDetected.shape[0]):
                keypointsPadded = kp2DDetected[iCam, :, None, :2]
                keypointsPaddedUndist = cv2.undistortPoints(keypointsPadded, s.intrinsicMats[iCam],
                                    s.undistParams[iCam], P=s.intrinsicMats[iCam])
                keypointsPaddedUndist = np.squeeze(keypointsPaddedUndist)
                s.kps2D.append(keypointsPaddedUndist)
        else:
            for iCam in range(kp2DDetected.shape[0]):
                s.kps2D.append(kp2DDetected[iCam, :, :2])

        # s.intrinsicMats = torch.tensor(s.intrinsicMats, dtype=dtype, requires_grad=True, device=s.device)
        s.projMats = torch.tensor(s.projMats, dtype=dtype, requires_grad=False, device=s.device)

        s.lossFunc = GMoF(GMRho)
        numAllJoints = 67
        numBodyJoints = 25
        headJointsId = [0, 15, 16, 17, 18]
        s.bodyJoints = [i for i in range(numBodyJoints) if i not in headJointsId]


    def forward(s, kps3D, normalize=False, noBodyJoint=True):
        weights = s.kp2DDetected[..., 2]
        weights[weights < s.minConfidence] = 0
        if noBodyJoint:
            weights[:, s.bodyJoints] = 0

        weights = torch.tensor(weights, dtype=s.dtype, requires_grad=False, device=s.device)

        kps2D = torch.tensor(s.kps2D, dtype=s.dtype, requires_grad=False, device=s.device)

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

        distance = torch.mean(s.lossFunc((kpProj2D - kps2D)) * weights[...,None])
        return distance

def loadKpJsonFile(kpFile, loadOnly1st=True):
    body_keypoints_multiPerson = []
    kpData = json.load(open(kpF))

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
    # kp2DMultiCams.append(body_keypoints_multiPerson[0])

    if loadOnly1st:
        return body_keypoints_multiPerson[0]
    else:
        return body_keypoints_multiPerson

if __name__ == '__main__':
    # kpFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03052\toRGB\KeypointsJson'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # # smplshExampleFile = r'..\Data\BuildSmplsh\Input\SMPLWithSocks_tri_Aligned.obj'
    # smplshExampleFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\03052.obj'
    # smplshDataFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    # outFolder = r'F:\WorkingCopy2\2020_07_09_TestSimplify\SimplifyLossFitting\03052'
    #
    # # test new smplsh data
    # inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    kpFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\toRGB\Undist\KeypointsJson'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # smplshExampleFile = r'..\Data\BuildSmplsh\Input\SMPLWithSocks_tri_Aligned.obj'
    smplshExampleFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\03067.obj'
    smplshDataFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    outFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067'
    targetSPCFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003067.obj'
    skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # test new smplsh data
    inFittingParam = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\Param_00499.npz'
    inPersonalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
    toSPCInterpoMatFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
    initializePose = False

    # device = torch.device("cuda:0")
    device = torch.device("cpu")
    # torch.cuda.set_device(device)
    learningRate = 5e-3
    # numIterations = 100000
    numIterations = 10000
    jointRegularizerWeight = 0.000001
    minKpConfidence = 0.5

    kpFiles = glob.glob(join(kpFolder, '*.json'))
    kpFiles.sort()

    kp2DMultiCams = []
    for kpF in kpFiles:
        # print(kpData)
        kp2DMultiCams.append(loadKpJsonFile(kpF))

    actual_img_shape = (2160, 4000)
    # camParams = Utility.load_cameras(camParamF, device, actual_img_shape)
    camParams=json.load(open(camParamF))
    camParams = [camParams['cam_params'][str(i)] for i in range(16)]


    smplshMesh = pv.PolyData(smplshExampleFile)
    smplshVerts = to_tensor(smplshMesh.points, device)
    personalShape = torch.tensor(np.load(inPersonalShapeFile)/1000, dtype=torch.float32, requires_grad=False, device=device)
    smplsh = smplsh_torch.SMPLModel(device, smplshDataFile, personalShape=personalShape)

    joints = smplsh.J_regressor.type(torch.float32) @ smplshVerts

    jointConverter = Utility.VertexToOpJointsConverter()
    opJoints = jointConverter(smplshVerts[None,...], joints[None,...])

    kp2DMultiCams = np.array(kp2DMultiCams)
    kp2DMultiCams[:, :, :2] = kp2DMultiCams[:, :, :2] * 2

    kpLoss = KeypointsLoss2D(device, camParams, kp2DMultiCams, undist2DKp=False, minConfidence=minKpConfidence)

    loss = kpLoss(opJoints[0, ...])
    # loss = kpLoss(opJoints[0, ...], kp2DMultiCams, undist2DKp=False)

    print('loss:', loss)

    # 3D to sparse point cloud loss
    targetSPCNP = pv.PolyData(targetSPCFile).points / 1000
    interpoMat = torch.tensor(np.load(toSPCInterpoMatFile), dtype=torch.float32, requires_grad=False, device=device)
    targetSPC = torch.tensor(targetSPCNP, dtype=torch.float32, requires_grad=False, device=device)
    skelData = json.load(open(skelDataFile))
    coarseMeshPts = np.array(skelData['VTemplate'])
    validVertsOnRestpose = np.where(coarseMeshPts[2, :] != -1)[0]
    obsIds = np.where(targetSPC[:, 2] > 0)[0]
    constraintIds = torch.tensor(np.intersect1d(obsIds, validVertsOnRestpose), dtype=torch.long, requires_grad=False, device=device)

    transInit, poseInit, betaInit = loadCompressedFittingParam(inFittingParam, readPersonalShape=False)
    if not initializePose:
        # transInit = -np.mean(targetSPCNP, axis=0)
        transInit = transInit * 0

        poseInit = poseInit * 0
    # Make fitting parameter tensors
    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=True, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=True, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=True, device=device)
    verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)

    optimizer = torch.optim.Adam([trans, pose], lr=learningRate)

    loop = tqdm.tqdm(range(numIterations))

    os.makedirs(outFolder, exist_ok=True)
    # main optimization loop
    for i in loop:
        optimizer.zero_grad()
        verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)
        verts = verts.type(torch.float32)
        jointsDeformed = jointsDeformed.type(torch.float32)
        opJoints = jointConverter(verts[None, ...], jointsDeformed[None, ...])

        kp2Dloss = kpLoss(opJoints[0, ...]*1000,)

        toSPCLoss = torch.mean(torch.index_select(targetSPC - interpoMat @ verts, 0, constraintIds)**2)

        jRegularzerLoss = jointRegularizerWeight * torch.sum(pose**2)

        loop.set_description('Step: ' + str(i) + ' kp2Dloss: ' + str(kp2Dloss.item()) + ' toSPCLoss: ' + str(toSPCLoss.item()) + ' jRegularzerLoss: ' + str(jRegularzerLoss))

        loss = 0.01*kp2Dloss + toSPCLoss + jRegularzerLoss
        # loss = toSPCLoss + jRegularzerLoss
        loss.backward()
        optimizer.step()

        if not (1+i) % 2000 and i:
            if i+1 == numIterations:
                smplsh.write_obj(verts * 1000, join(outFolder, 'TpSparseFinal.obj'))
            else:
                smplsh.write_obj(verts * 1000, join(outFolder, 'Opt_' + str(i).zfill(5) + '.obj'))


    outParamFile = join(outFolder, 'ToSparseFittingParams.npz')
    np.savez(outParamFile, trans=trans.detach().numpy(), pose = pose.detach().numpy(), beta=betas.detach().numpy(), personalShape=personalShape.detach().numpy())