import sys
import os
import glob
sys.path.append(os.path.abspath(''))

from datetime import datetime

SMPLSH_Dir = r'..\SMPL_reimp'

import sys
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch


from os.path import join
import pyvista as pv
import torch
print(torch.version.cuda)


import numpy as np
from Utility import *

if __name__ == '__main__':
    # fitParamFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\XYZ_RestPose_HHFix_Sig_1e-07_BR1e-07_Fpp6_NCams16ImS1080_LR0.4_LW1e-07_NW0.0_Batch4\FitParam'
    # smplshData = r'..\SMPL_reimp\SmplshModel_m.npz'
    # smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
    # personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'
    # # interpolatedMeshFile =  r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolatedWithSparse.ply'
    # interpolatedMeshFile =  r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\FinalMesh.obj'

    fitParamFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\Param_00499.npz'
    smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
    outPersonalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
    outBetaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'
    # interpolatedMeshFile =  r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolatedWithSparse.ply'
    # The mesh after to sparse interpolation, NOT the mesh after to silhouette fitting
    interpolatedMeshFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolatedWithSparse.ply'
    interpolatedMeshObjFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolatedWithSparse.obj'

    device = torch.device("cuda:0")
    smplsh = smplsh_torch.SMPLModel(device, smplshData, personalShape=None, unitMM=True)

    # fitParamFiles= glob.glob(join(fitParamFolder, '*.npz'))
    # fitParamFiles.sort()
    pose_size = 3 * 52

    smplshExampleMesh = pv.PolyData(smplshExampleMeshFile)

    param = np.load(fitParamFile)
    personalShapeFinal = param['personalShape']
    trans = param['trans']
    pose = param['pose']
    beta = param['beta']

    pose = torch.tensor(pose, dtype=torch.float64, requires_grad=False, device=device)
    beta = torch.tensor(beta, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(trans, dtype=torch.float64,
                         requires_grad=False, device=device)

    T, pbs, v_shaped = smplsh.getTransformation(beta, pose, trans, returnPoseBlendShape=True)

    inverseTransform = np.zeros(T.shape, dtype=np.float64)

    interpolatedMesh = pv.PolyData(interpolatedMeshFile)
    interpolatedVerts = np.array(interpolatedMesh.points)
    personalShapeFinalRestpose = np.zeros(interpolatedVerts.shape, dtype=np.float64)

    for i in range(T.shape[0]):
        inverseTransform[i, :, :] = np.linalg.inv(T[i, :, :].cpu().detach().numpy())
        pt = interpolatedVerts[i:i+1, :].transpose()
        pt = np.vstack([pt, 1])

        ptBackToRest = inverseTransform[i, :, :] @ pt
        personalShapeFinalRestpose[i, :] = ptBackToRest[:3, 0]

    # the rest pose has also been applied with pose blend shape, we need to deduct it
    personalShapeFinalRestpose = personalShapeFinalRestpose - pbs.cpu().detach().numpy()

    # then get the pure smplsh rest pose shape

    interpolatedMesh.points = personalShapeFinalRestpose
    interpolatedMesh.save(join(r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final', 'DisplacementToRestpose.ply'))

    np.save(outPersonalShapeFile, personalShapeFinalRestpose-v_shaped.cpu().detach().numpy())
    np.save(outBetaFile, beta.cpu().numpy())