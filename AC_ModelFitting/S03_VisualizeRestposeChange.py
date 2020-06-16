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
    fitParamFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\XYZ_RestPose_HHFix_Sig_1e-07_BR1e-07_Fpp6_NCams16ImS1080_LR0.4_LW1e-07_NW0.0_Batch4\FitParam'
    smplshData = r'..\SMPL_reimp\SmplshModel_m.npz'
    smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
    outFolderMesh = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\RealDataSilhouette\XYZ_RestPose_HHFix_Sig_1e-07_BR1e-07_Fpp6_NCams16ImS1080_LR0.4_LW1e-07_NW0.0_Batch4\RestposeChange'
    personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'

    device = torch.device("cuda:0")
    smplsh = smplsh_torch.SMPLModel(device, smplshData, personalShape=None, unitMM=True)

    fitParamFiles= glob.glob(join(fitParamFolder, '*.npz'))
    pose_size = 3 * 52

    smplshExampleMesh = pv.PolyData(smplshExampleMeshFile)

    os.makedirs(outFolderMesh, exist_ok=True)

    for i, paramF in enumerate(fitParamFiles):
        param = np.load(paramF)
        personalShape = torch.tensor(param['personalShape'], dtype=torch.float64, requires_grad=True, device=device)
        betas = param['beta']

        pose = torch.tensor(np.zeros((pose_size,)), dtype=torch.float64, requires_grad=True, device=device)
        betas = torch.tensor(betas, dtype=torch.float64, requires_grad=True, device=device)
        trans = torch.tensor([0,0,0], dtype=torch.float64,
                             requires_grad=True, device=device)

        smplsh.personalShape = personalShape

        with torch.no_grad():
            verts = smplsh(betas, pose, trans).type(torch.float32)

        saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

    param = np.load(fitParamFiles[-1])
    personalShapeFinal = param['personalShape']
    np.save(personalShapeFile, personalShapeFinal)