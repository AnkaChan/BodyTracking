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
from S05_InterpolateWithSparsePointCloud import *
from S07_ToSilhouetteFitting_MultiFrames import *

import numpy as np
from Utility import *

def getPersonalShapeFromInterpolation(inMeshFile, inSparseCloud, inFittingParamFile, outInterpolatedFile, outFittingParamFileWithPS,
    skelDataFile, interpoMatFile, laplacianMatFile=None, smplshData=r'..\SMPL_reimp\SmplshModel_m.npz',\
    handIndicesFile = r'HandIndices.json', HeadIndicesFile = 'HeadIndices.json', softConstraintWeight = 100,
    numRealCorners = 1487, fixHandAndHead = True, ):

    interpolateWithSparsePointCloudSoftly(inMeshFile, inSparseCloud, outInterpolatedFile, skelDataFile,
                                         interpoMatFile, laplacianMatFile=laplacianMatFile, \
                                         handIndicesFile=handIndicesFile, HeadIndicesFile=HeadIndicesFile,
                                         softConstraintWeight=softConstraintWeight,
                                         numRealCorners=numRealCorners, fixHandAndHead=fixHandAndHead)
    device = torch.device("cuda:0")
    smplsh = smplsh_torch.SMPLModel(device, smplshData, personalShape=None, unitMM=True)

    param = np.load(inFittingParamFile)
    # personalShapeFinal = param['personalShape']
    trans = param['trans'] * 1000
    pose = param['pose']
    beta = param['beta']

    pose = torch.tensor(pose, dtype=torch.float64, requires_grad=False, device=device)
    beta = torch.tensor(beta, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(trans, dtype=torch.float64,
                         requires_grad=False, device=device)

    verts = smplsh(beta, pose, trans)
    smplsh.write_obj(verts, 'SmplshDeformation.obj')

    T, pbs, v_shaped = smplsh.getTransformation(beta, pose, trans, returnPoseBlendShape=True)

    inverseTransform = np.zeros(T.shape, dtype=np.float64)

    interpolatedMesh = pv.PolyData(outInterpolatedFile)
    interpolatedVerts = np.array(interpolatedMesh.points)
    personalShapeFinalRestpose = np.zeros(interpolatedVerts.shape, dtype=np.float64)

    for i in range(T.shape[0]):
        inverseTransform[i, :, :] = np.linalg.inv(T[i, :, :].cpu().detach().numpy())
        pt = interpolatedVerts[i:i + 1, :].transpose()
        pt = np.vstack([pt, 1])

        ptBackToRest = inverseTransform[i, :, :] @ pt
        personalShapeFinalRestpose[i, :] = ptBackToRest[:3, 0]

    # the rest pose has also been applied with pose blend shape, we need to deduct it
    personalShapeFinalRestpose = (personalShapeFinalRestpose- pbs.cpu().detach().numpy())/1000

    # then get the pure smplsh rest pose shape
    np.savez(outFittingParamFileWithPS, trans=trans.cpu().numpy()/1000, pose=pose.cpu().numpy(), beta=beta.cpu().numpy(),
             personalShape=personalShapeFinalRestpose-v_shaped.cpu().numpy()/1000)

    interpolatedMesh = pv.PolyData(outInterpolatedFile)

    interpolatedMesh.points = personalShapeFinalRestpose
    interpolatedMesh.save('DisplacementToRestpose.ply')

if __name__ == '__main__':
    sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06950\A00006950.obj'
    initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\06950.npz'
    inMeshFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\06950.obj'
    laplacianMatFile = r'SmplshRestposeLapMat.npy'
    outFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\Model\06950'
    skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    interpoMat = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolationMatrix.npy'

    os.makedirs(outFolder, exist_ok=True)
    outInterpolatedFile = join(outFolder, 'Interpolated.ply')
    outFittingParamFileWithPS = join(outFolder, 'FitParams.npz')

    getPersonalShapeFromInterpolation(inMeshFile, sparsePointCloudFile,initialFittingParamFile, outInterpolatedFile, \
                                      outFittingParamFileWithPS, skelDataFile, interpoMat, laplacianMatFile=laplacianMatFile)