from M04_TexturedFitting import *
class InputBundle:
    def __init__(s):
        # same over all frames
        s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.smplshRegressorMatFile = r'C:\Code\MyRepo\ChbCapture\08_CNNs\Openpose\SMPLSHAlignToAdamWithHeadNoFemurHead\smplshRegressorNoFlatten.npy'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
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
    inputs = InputBundle()

    # inputs.imageFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\toRGB\Undist'
    inputs.imageFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\Preprocessed\03067'
    # inputs.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
    inputs.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003067.obj'
    inputs.cleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    inputs.compressedStorage = True
    # inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\06950.npz'
    inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr\ToSparseFittingParams.npz'
    inDeformedMeshFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr\ToSparseFitFinalMesh.obj'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr'
    inputs.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
    inputs.texturedMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TestMultiResolution\03067'

    cfg = RenderingCfg()
    cfg.sigma = 1e-7
    cfg.blurRange = 1e-7

    cfg.numIterations = 2000
    cfg.plotStep = 1000
    cfg.numCams = 16
    # low learning rate for pose optimization
    cfg.learningRate = 1e-4

    cfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    # cfg.imgSize = 1080
    # cfg.batchSize = 2

    # cfg.imgSize = 540
    cfg.imgSize = 256
    cfg.batchSize = 8

    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    # cfg.numIterations = 20
    cfg.useKeypoints = False
    cfg.kpFixingWeight = 0
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    # cfg.bin_size = 256
    cfg.bin_size = None
    cfg.inputImgExt = 'png'
    cfg.terminateStep = 1e-6

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    texturedPoseFitting(inputs, cfg, device)