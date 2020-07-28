from S12_RealData_TexturePerVertexFitting import *

class InputBundle:
    def __init__(s):
        # same over all frames
        s.camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\SMPLSH.obj'
        s.toSparsePCMat = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.smplshRegressorMatFile = r'smplshRegressorNoFlatten.npy'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"
        s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

        # frame specific inputs
        s.imageFolder = r'F:\WorkingCopy2\2020_06_04_SilhouetteExtraction\03067\silhouettes'
        s.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
        s.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\A00003067.obj'

        s.compressedStorage = True
        s.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03067.npz'
        s.outputFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\03067'
        # copy all the final result to this folder
        s.finalOutputFolder = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final'

if __name__ == '__main__':
    cfg = RenderingCfg()

    # cfg.sigma = 0
    # cfg.blurRange = 0
    cfg.sigma = 1e-8
    cfg.blurRange = 1e-8

    # cfg.plotStep = 5
    cfg.numCams = 16
    # low learning rate for pose optimization
    # cfg.learningRate = 2e-3
    # cfg.learningRate = 1e-2
    cfg.learningRate = 1e-2
    # cfg.learningRate = 1
    # cfg.learningRate = 100

    cfg.batchSize = 4
    # cfg.faces_per_pixel = 10 # for testing
    cfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    # cfg.lpSmootherW = 1e-10
    # cfg.lpSmootherW = 1e-4
    cfg.lpSmootherW = 1e-1
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 5000
    # cfg.numIterations = 20
    cfg.useKeypoints = False
    cfg.kpFixingWeight = 0
    cfg.noiseLevel = 0.1
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    cfg.toSparseCornersFixingWeight = 1
    cfg.ambientLvl = 0.8
    cfg.terminateStep = 1e-8

    # cfg.plotStep = cfg.numIterations
    cfg.plotStep = 200
    cfg.drawInitial = False
    cfg.undistImg = False
    cfg.inputImgExt = 'png'

    cfg.bin_size = 128
    pose_size = 3 * 52
    beta_size = 10

    inputs = InputBundle()

    # inputs.imageFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06950'
    # # inputs.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
    # inputs.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06950\A00006950.obj'
    # inputs.cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted\Undist'
    # inputs.compressedStorage = True
    # inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_21_TextureRendering\Model\06950\Param_00959.npz'
    # inputs.outputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\RealDataPerVertsFitting\06950'

    inputs.imageFolder = r'Z:\ShareZ\2020_05_21_AC_FramesDataToFitTo\Copied\03067\toRGB\Undist'
    # inputs.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
    inputs.sparsePointCloudFile = r'Z:\ShareZ\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003067.obj'
    inputs.cleanPlateFolder = r'Z:\shareZ\2020_07_15_NewInitialFitting\CleanPlatesExtracted\rgb\Undist'
    inputs.compressedStorage = True
    inputs.initialFittingParamFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067\PoseParam_00077.npz'
    inputs.outputFolder = r'Z:\shareZ\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067'
    inputs.texturedMesh = r'..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj'

    # copy all the final result to this folder
    # inputs.finalOutputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\RealDataPoseFitting'
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    texturedPerVertexFitting(inputs, cfg, device)










