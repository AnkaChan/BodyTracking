# from S01_Build_SMPL_Socks import *

from S04_SMPLSH_FitToSparserWithOPKeypoints import *

if __name__ == '__main__':

    SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # outFolder = r'SMPLSHFit\LadaOldSuit_WithOPKeypoints'
    outFolderAll = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting'

    deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed'
    inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
    # inputDensePointCloudFile = r'F:\WorkingCopy2\2020_04_05_LadaRestPosePointCloud\Pointclouds\03052\scene_dense.ply'
    inputDensePointCloudFile = None
    inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'
    toSparsePCMat = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolationMatrix.npy'
    personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
    betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'

    cfg = Config()
    cfg.jointRegularizerWeight = 0.000001
    # jointRegularizerWeight = 0.0000001
    # jointRegularizerWeight = 0
    cfg.learnrate_ph = 0.01
    cfg.lrDecayStep = 100
    cfg.lrDecayRate = 0.97
    # numIterToKp = 3000
    # printStep = 500

    # noBodyKeyJoint = False
    cfg.noBodyKeyJoint = True
    cfg.numBodyJoint = 25
    cfg.headJointsId = [0, 15, 16, 17, 18]

    cfg.numIterFitting = 5000
    cfg.printStep = 100

    cfg.indicesVertsToOptimize = list(range(6750))

    # keypointFitWeightInToDenseICP = 0.1
    cfg.withDensePointCloud = False
    # cfg.keypointFitWeightInToDenseICP = 1
    # keypointFitWeightInToDenseICP = 0.0
    cfg.constantBeta = True
    cfg.betaRegularizerWeightToKP = 0
    cfg.manualCorrsWeightToKP = 1

    toSparseFittingNewRegressor(inputKeypoints, targetMesh, outFolder, skelDataFile, toSparsePCMat, betaFile, personalShapeFile, SMPLSHNpzFile, cfg=cfg)


