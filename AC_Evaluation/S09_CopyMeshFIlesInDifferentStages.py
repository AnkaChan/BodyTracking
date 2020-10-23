import sys
sys.path.append('../AC_FitPipeline')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Utility import *
import shutil
import M03_ToSparseFitting

if __name__ == '__main__':
    toSparseFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    finalFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Final\Mesh'
    outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation'
    kpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'
    densePCFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Dense'
    frames = [str(frameId) for frameId in range(10459, 10459 + 300)]
    # frames = [str(i).zfill(5) for i in range(8564, 8564 + 300)]

    cfg = M03_ToSparseFitting.Config()
    # cfg.learnrate_ph = 0.05
    cfg.learnrate_ph = 0.01
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.005
    cfg.lrDecayStep = 200
    cfg.lrDecayRate = 0.96
    cfg.numComputeClosest = 5
    cfg.numIterFitting = 100
    cfg.noBodyKeyJoint = False
    cfg.betaRegularizerWeightToKP = 1
    cfg.outputErrs = True
    cfg.constantBeta = False
    # cfg.betaRegularizerWeightToKP = 0.1
    # cfg.jointRegressor = jointRegularizerWeight = 1e-5
    cfg.withDensePointCloud = True
    cfg.terminateLossStep = 1e-9
    cfg.maxDistanceToClosestPt = 0.05

    doCopyFile = False
    # doCopyFile = True


    interpolationFolder = join(outFolder, "Interpolated")
    ImageBasedFittingFolder = join(outFolder, 'ImageBasedFitting')
    toTrackingPointsFolder = join(outFolder, 'ToTrackingPoints')
    toDenseFolder = join(outFolder, 'toDense')

    os.makedirs(interpolationFolder, exist_ok=True)
    os.makedirs(ImageBasedFittingFolder, exist_ok=True)
    os.makedirs(toTrackingPointsFolder, exist_ok=True)
    os.makedirs(toDenseFolder, exist_ok=True)

    for frameName in frames:
        print("Processing Frame: ", frameName)
        toSparseProcessedFolder = join(toSparseFolder, frameName)

        finalMesh = join(finalFolder, 'A' + frameName + '.ply')

        toTPFile = join(toSparseProcessedFolder, 'ToSparseMesh.obj')
        interpolatedFile = join(toSparseProcessedFolder, 'InterpolatedMesh.obj')

        if doCopyFile:
            shutil.copy(finalMesh, join(ImageBasedFittingFolder, 'A' + frameName + '.ply'))
            shutil.copy(toTPFile, join(toTrackingPointsFolder, 'A' + frameName + '.obj'))
            shutil.copy(interpolatedFile, join(interpolationFolder, 'A' + frameName + '.obj'))

        inputKeypoints = join(kpFolder, frameName + '.obj')
        betaFile = None
        personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        smplshDataFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        initialPoseFile = join(toSparseProcessedFolder, 'ToSparseFittingParams.npz')
        densePointCloudFile = join(densePCFolder, frameName + '.ply')

        runningFolder = join(toDenseFolder, 'Running', frameName)
        os.makedirs(runningFolder, exist_ok=True)

        M03_ToSparseFitting.toSparseFittingKeypoints(inputKeypoints, runningFolder, betaFile, personalShapeFile,
                                                     smplshDataFile, initialPoseFile=initialPoseFile,
                                                     inputDensePointCloudFile=densePointCloudFile, cfg=cfg)

        shutil.copy( join(runningFolder, r'ToSparseMesh.obj'), join(toDenseFolder, 'A' + frameName + '.obj'))

