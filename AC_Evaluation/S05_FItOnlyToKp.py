import sys
sys.path.append('../AC_FitPipeline')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import M05_Visualization
import M03_ToSparseFitting
import Utility

if __name__ == '__main__':
    cfg = M03_ToSparseFitting.Config()
    cfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.005
    cfg.lrDecayStep = 200
    cfg.lrDecayRate = 0.96
    cfg.numComputeClosest = 10
    cfg.numIterFitting = 300
    cfg.noBodyKeyJoint = False
    cfg.betaRegularizerWeightToKP = 1000
    cfg.outputErrs = True
    cfg.constantBeta = False
    # cfg.betaRegularizerWeightToKP = 0.1
    # cfg.jointRegressor = jointRegularizerWeight = 1e-5
    cfg.withDensePointCloud = True
    cfg.terminateLossStep = 1e-9
    cfg.maxDistanceToClosestPt = 0.1

    inputKeypoints = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints\10459.obj'
    outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ToKpAndDense\10459'
    smplshDataFile =  r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    sparsePCObjFile = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LadaStand\A00010459.obj'
    toSparsePointCloudInterpoMatFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
    skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    densePointCloudFile = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Dense\10459.ply'
    initialPoseFile = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ToKpOnly\10459\ToSparseFittingParams.npz'

    betaFile = None
    personalShapeFile = None

    # only to keypoints fails and not necessary.
    # M03_ToSparseFitting.toSparseFittingKeypoints(inputKeypoints, outFolder, betaFile, personalShapeFile, smplshDataFile,
    #                          initialPoseFile=None, cfg=cfg)

    # M03_ToSparseFitting.toSparseFittingNewRegressor(inputKeypoints, sparsePCObjFile, outFolder, skelDataFile, toSparsePointCloudInterpoMatFile,
    #                 betaFile, personalShapeFile, smplshDataFile, densePointCloudFile = densePointCloudFile, cfg=cfg)

    M03_ToSparseFitting.toSparseFittingKeypoints(inputKeypoints, outFolder, betaFile, personalShapeFile, smplshDataFile,
                             initialPoseFile=initialPoseFile,  inputDensePointCloudFile = densePointCloudFile, cfg=cfg)