from S03_TexturedFittingWithSIlhouetteForSelectedFrames import *

class InputBundle():
    def __init__(s):

        s.camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\SMPLSH.obj'
        s.toSparsePCMat = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.inputDensePointCloudFile = None
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'
        s.texturedMesh = "..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj"
        s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.cleanPlateFolder = r'Z:\shareZ\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
        s.compressedStorage = True

        # frame specific inputs
        s.dataFolder = r'Z:\shareZ\2020_07_26_NewPipelineTestData'
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = join(s.dataFolder, 'Keypoints')
        s.toSparseFittedFolder = None
        s.outputFolderAll = None
        s.initialFittingParamFile = None

if __name__ == '__main__':
    inputs = InputBundle()
    frameNames = [
                # '03067',
                  # '03990',
                  # '04735',
                  # '04917',
                  '06250',
                    '06550',
                  # '06950'
                  ]

    inputs.inOriginalObjFilesFolder = r'Z:\shareZ\2020_05_21_AC_FramesDataToFitTo\Copied\ObjFiles'
    inputs.toSparseFittedFolder = r'Z:\shareZ\2020_07_26_NewPipelineTestData\ToSparse'
    inputs.silhouetteFolderParent =r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting'
    inputs.texturedMesh = r'..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj'
    # inputs.outputFolderAll = r'Z:\shareZ\2020_07_26_NewPipelineTestData\TexturedFitting'
    # inputs.outputFolderAll = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\TexturedFitting_Size540'
    inputs.outputFolderAll = r'Z:\shareZ\2020_07_26_NewPipelineTestData\TexturedFittingWithSilhouettes'
    inputs.outputFolderFinal = r'Z:\shareZ\2020_07_26_NewPipelineTestData\TextureCompletionWithSilhouettes'

    cfg = Config()
    # cfg.skipPerVertFitting = True
    # cfg.skipPoseFitting = True

    cfg.texturedPoseFittingCfg.sigma = 1e-7
    cfg.texturedPoseFittingCfg.blurRange = 1e-7

    cfg.texturedPoseFittingCfg.plotStep = 100
    cfg.texturedPoseFittingCfg.numCams = 16
    # low learning rate for pose optimization
    cfg.texturedPoseFittingCfg.learningRate = 1e-4

    cfg.texturedPoseFittingCfg.faces_per_pixel = 2  # for debugging
    # cfg.imgSize = 2160
    # cfg.texturedPoseFittingCfg.imgSize = 1080
    # cfg.texturedPoseFittingCfg.batchSize = 2

    # cfg.texturedPoseFittingCfg.imgSize = 1080
    cfg.texturedPoseFittingCfg.imgSize = 540
    cfg.texturedPoseFittingCfg.batchSize = 2

    cfg.texturedPoseFittingCfg.terminateLoss = 0.1
    cfg.texturedPoseFittingCfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.texturedPoseFittingCfg.normalSmootherW = 0.0
    cfg.texturedPoseFittingCfg.numIterations = 500
    # cfg.numIterations = 20
    cfg.texturedPoseFittingCfg.useKeypoints = False
    cfg.texturedPoseFittingCfg.kpFixingWeight = 0
    cfg.texturedPoseFittingCfg.noiseLevel = 0.1
    cfg.texturedPoseFittingCfg.bodyJointOnly = True
    cfg.texturedPoseFittingCfg.jointRegularizerWeight = 1e-5
    cfg.texturedPoseFittingCfg.bin_size = None
    cfg.texturedPoseFittingCfg.inputImgExt = 'png'
    # cfg.texturedPoseFittingCfg.terminateStep = 1e-6
    cfg.texturedPoseFittingCfg.terminateStep = 1e-7
    cfg.texturedPoseFittingCfg.withSilhouette = True
    cfg.texturedPoseFittingCfg.silhouetteLossWeight = 0.5
    cfg.texturedPoseFittingCfg.extrinsicsOutsideCamera = True

    cfg.texturedPerVertexFittingCfg.sigma = 1e-8
    cfg.texturedPerVertexFittingCfg.blurRange = 1e-8

    cfg.texturedPerVertexFittingCfg.plotStep = 100
    # cfg.texturedPerVertexFittingCfg.plotStep = 20
    cfg.texturedPerVertexFittingCfg.numCams = 16
    cfg.texturedPerVertexFittingCfg.learningRate = 1e-4
    cfg.texturedPerVertexFittingCfg.faces_per_pixel = 2  # for debugging
    # cfg.texturedPerVertexFittingCfg.imgSize = 1080
    # cfg.texturedPerVertexFittingCfg.batchSize = 2

    cfg.texturedPerVertexFittingCfg.imgSize = 1080
    cfg.texturedPerVertexFittingCfg.batchSize = 2

    cfg.texturedPerVertexFittingCfg.terminateLoss = 0.1
    cfg.texturedPerVertexFittingCfg.lpSmootherW = 1e-2
    cfg.texturedPerVertexFittingCfg.normalSmootherW = 0.0
    cfg.texturedPerVertexFittingCfg.numIterations = 500
    cfg.texturedPerVertexFittingCfg.useKeypoints = False
    cfg.texturedPerVertexFittingCfg.kpFixingWeight = 0
    cfg.texturedPerVertexFittingCfg.noiseLevel = 0.1
    cfg.texturedPerVertexFittingCfg.bodyJointOnly = True
    cfg.texturedPerVertexFittingCfg.jointRegularizerWeight = 1e-5
    cfg.texturedPerVertexFittingCfg.toSparseCornersFixingWeight = 1
    cfg.texturedPerVertexFittingCfg.bin_size = 256
    cfg.texturedPerVertexFittingCfg.inputImgExt = 'png'
    cfg.texturedPerVertexFittingCfg.drawInitial = True
    # cfg.texturedPerVertexFittingCfg.drawInitial = False
    # cfg.texturedPerVertexFittingCfg.optimizerType = 'SGD'
    cfg.texturedPerVertexFittingCfg.optimizerType = 'Adam'
    cfg.texturedPerVertexFittingCfg.terminateStep = 1e-7
    cfg.texturedPerVertexFittingCfg.withSilhouette = True
    cfg.texturedPerVertexFittingCfg.silhouetteLossWeight = 0.5
    cfg.texturedPerVertexFittingCfg.extrinsicsOutsideCamera = True


    texturedFittingWithSilhouettes(inputs, frameNames, cfg)