import numpy as np

import sys, os
from os.path import  join
import json
import math
import numpy as np
import SkelFit.Data
import glob,tqdm
import pyvista as pv
from pathlib import Path
from matplotlib import pyplot as plt
from M01_LBSFitting import  makeRegualrTemporalLaplacainBlock

def temporalSmoothingPointTrajectory(inFolder, outFolder, softConstraintWeight=100, inExt ='ply', outExt='ply'):
    # outFolder = join(outFolder, 'Final_Smoothed_Weight' + str(softConstraintWeight))
    os.makedirs(outFolder, exist_ok=True)
    vtkFiles = glob.glob(join(inFolder, '*.'+inExt))

    seqLength = len(vtkFiles)

    ptsTrajectory = []
    models = []
    for vtkF in vtkFiles:
        model = pv.PolyData(vtkF)
        models.append(model)
        # model.cell_normals = np.zeros(model.cell_normals.shape)

        # outModelWithNormals = join(outFolder, modelName + '.vtk')
        # model.save(outModelWithNormals)
        ptsTrajectory.append(model.points)

    ptsTrajectory = np.array(ptsTrajectory)
    tL1 = makeRegualrTemporalLaplacainBlock(1).toarray()

    print(tL1)

    sizeTLMat = ptsTrajectory.shape[0]
    tL = np.zeros((sizeTLMat, sizeTLMat))

    for i in tqdm.tqdm(range(1, sizeTLMat)):
        tL[i - 1:i + 1, i - 1:i + 1] += tL1

    print(tL.shape)

    S = np.eye(sizeTLMat)
    # p = np.array(nmTrajectory)
    # p.resize(sizeTLMat, 1)

    # nmTrajectorySmoothed = np.linalg.solve(tL + softConstraintWeight * np.transpose(S) @ S, 2*softConstraintWeight* np.transpose(S)@ p)
    # nmTrajectorySmoothed = np.linalg.solve(tL + softConstraintWeight * np.transpose(S) @ S, softConstraintWeight* np.transpose(S)@ p)
    A = tL + softConstraintWeight * np.transpose(S) @ S

    for iV in tqdm.tqdm(range(ptsTrajectory.shape[1]), desc='Solving smoothed trajectory'):
        pTrajectory = ptsTrajectory[:, iV, :]
        pTrajectorySmoothed = np.linalg.solve(A, softConstraintWeight * np.transpose(S) @ pTrajectory)

        # fig, axs = plt.subplots()
        # axs.plot(pTrajectory[:, 0], label="Original")
        # axs.plot(pTrajectorySmoothed[:, 0], label="Smoothed")
        # axs.legend()
        # axs.set_title('Normal trajectory')
        # plt.show()
        # plt.waitforbuttonpress(0)

        ptsTrajectory[:, iV, :] = pTrajectorySmoothed

    for iF in range(len(vtkFiles)):
        vtkF = vtkFiles[iF]
        modelName = Path(vtkF).stem
        outModelSmoothed = join(outFolder, modelName + '.'+outExt)

        model = models[iF]
        model.points = ptsTrajectory[iF, :, :]

        model.save(outModelSmoothed)