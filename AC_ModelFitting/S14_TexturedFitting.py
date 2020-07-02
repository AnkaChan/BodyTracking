from S11_RealData_TexturePoseFitting import *
from S12_RealData_TexturePerVertexFitting import *
from S13_GetPersonalShapeFromInterpolation import getPersonalShapeFromInterpolation
import copy

class Config:
    def __init__(s):
        s.texturedPoseFittingCfg = RenderingCfg()
        s.texturedPerVertexFittingCfg = RenderingCfg()


class InputBundle:
    def __init__(s):
        # person specific
        s.toSparsePointCloudInterpoMatFile = r'..\Data\PersonalModel_Lada\InterpolationMatrix.npy'
        # s.betaFile = r'..\Data\PersonalModel_Lada\Beta.npy'
        # s.personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'
        #
        # s.OP2AdamJointMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\OP2AdamJointMat.npy'
        # s.AdamGoodJointsFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\AdamGoodJoints.npy'
        # s.smplsh2OPRegressorMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\smplshRegressorNoFlatten.npy'
        # s.smplshDataFile = r'..\SMPL_reimp\SmplshModel_m.npz'

        s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolationMatrix.npy'
        s.smplshRegressorMatFile = r'C:\Code\MyRepo\ChbCapture\08_CNNs\Openpose\SMPLSHAlignToAdamWithHeadNoFemurHead\smplshRegressorNoFlatten.npy'
        s.smplshData = r'..\SMPL_reimp\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'
        s.texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"

        s.laplacianMatFile = r'SmplshRestposeLapMat.npy'
        s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted\Undist'

def texturedFitting(inputs, cfg=Config()):
    pose_size = 3 * 52
    beta_size = 10

    inImgFolders = glob.glob(join(inputs.inImgParentFolder, '*'))
    inCompleteObjFiles = glob.glob(join(inputs.inCompleteObjFilesFolder, '*.obj'))
    inOriginalObjFiles = glob.glob(join(inputs.inOriginalObjFilesFolder, '*.obj'))
    inImgFolders.sort()
    inCompleteObjFiles.sort()
    inOriginalObjFiles.sort()

    poseFittingFolder =join(inputs.outputFolder, 'PoseFitting')
    os.makedirs(poseFittingFolder, exist_ok=True)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    for inImgFolder, inOriginalObjFile,  inCompleteObjFile in zip(inImgFolders, inOriginalObjFiles, inCompleteObjFiles):
        # First interpolated to completed sparse mesh, and get accurate personal rest pose shape
        frameName = os.path.basename(inImgFolder)
        sparsePointCloudFile = inCompleteObjFile

        initialFittingParamFile = join(inputs.toSparseFittedFolder, frameName, 'FittingParams', frameName + '.npz')
        inDeformedMeshFile = join(inputs.toSparseFittedFolder, frameName, frameName + '.obj')

        outFolderInterpolation = join(poseFittingFolder, 'Interpolation', frameName)
        os.makedirs(outFolderInterpolation, exist_ok=True)
        outInterpolatedFile = join(outFolderInterpolation, frameName + 'Interpolation.ply')
        outFittingParamFileWithPS = join(outFolderInterpolation, frameName + 'WithAccuratePS.npz')

        # getPersonalShapeFromInterpolation(inDeformedMeshFile, sparsePointCloudFile, initialFittingParamFile, outInterpolatedFile,
        #                                       outFittingParamFileWithPS,
        #                                       inputs.skelDataFile, inputs.toSparsePointCloudInterpoMatFile)

        # textured pose fitting
        inputsPoseFitting = copy.copy(inputs)
        inputsPoseFitting.imageFolder = inImgFolder
        inputsPoseFitting.KeypointsFile = join(inImgFolder, 'oRGB', 'Reconstruction', 'PointCloud.obj')
        inputsPoseFitting.compressedStorage = True
        inputsPoseFitting.sparsePointCloudFile = inOriginalObjFile
        inputsPoseFitting.initialFittingParamFile = outFittingParamFileWithPS
        inputsPoseFitting.outputFolder = join(poseFittingFolder, 'Fitting', frameName)
        os.makedirs(inputsPoseFitting.outputFolder, exist_ok=True)

        # texturedPoseFitting(inputsPoseFitting, cfg.texturedPoseFittingCfg, device)

        poseFittingParamFolder, _ = makeOutputFolder(inputsPoseFitting.outputFolder, cfg.texturedPoseFittingCfg, Prefix='Pose_')
        paramFiles = glob.glob(join(poseFittingParamFolder, 'FitParam', '*.npz'))
        paramFiles.sort()
        finalPoseFile = paramFiles[-1]

        # textured per vertex fitting
        inputsPerVertexFitting = copy.copy(inputsPoseFitting)
        inputsPerVertexFitting.imageFolder = join(inImgFolder, 'Undist')
        inputsPerVertexFitting.compressedStorage = True
        inputsPerVertexFitting.initialFittingParamFile = finalPoseFile

        inputsPerVertexFitting.outputFolder = join(inputs.outputFolder, 'PerVertexFitting', frameName)
        os.makedirs(inputsPerVertexFitting.outputFolder, exist_ok=True)

        texturedPerVertexFitting(inputsPerVertexFitting, cfg.texturedPerVertexFittingCfg, device)

if __name__ == '__main__':
    inputs = InputBundle()
    inputs.inImgParentFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\Copied\Images'
    inputs.inCompleteObjFilesFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_0_JBiLap_0_Step10_Overlap0\Deformed'
    inputs.inOriginalObjFilesFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\Copied\ObjFiles'
    inputs.toSparseFittedFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\ToSparse'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting'

    cfg = Config()
    cfg.texturedPoseFittingCfg.sigma = 1e-8
    cfg.texturedPoseFittingCfg.blurRange = 1e-8

    cfg.texturedPoseFittingCfg.plotStep = 50
    cfg.texturedPoseFittingCfg.numCams = 16
    # low learning rate for pose optimization
    cfg.texturedPoseFittingCfg.learningRate = 5e-5

    cfg.texturedPoseFittingCfg.batchSize = 2
    cfg.texturedPoseFittingCfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    cfg.texturedPoseFittingCfg.imgSize = 1080
    cfg.texturedPoseFittingCfg.terminateLoss = 0.1
    cfg.texturedPoseFittingCfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.texturedPoseFittingCfg.normalSmootherW = 0.0
    cfg.texturedPoseFittingCfg.numIterations = 200
    # cfg.numIterations = 20
    cfg.texturedPoseFittingCfg.useKeypoints = False
    cfg.texturedPoseFittingCfg.kpFixingWeight = 0
    cfg.texturedPoseFittingCfg.noiseLevel = 0.1
    cfg.texturedPoseFittingCfg.bodyJointOnly = True
    cfg.texturedPoseFittingCfg.jointRegularizerWeight = 1e-5
    cfg.texturedPoseFittingCfg.bin_size = 256

    cfg.texturedPerVertexFittingCfg.sigma = 1e-8
    cfg.texturedPerVertexFittingCfg.blurRange = 1e-8

    cfg.texturedPerVertexFittingCfg.plotStep = 50
    cfg.texturedPerVertexFittingCfg.numCams = 16
    cfg.texturedPerVertexFittingCfg.learningRate = 1e-4
    cfg.texturedPerVertexFittingCfg.batchSize = 2
    cfg.texturedPerVertexFittingCfg.faces_per_pixel = 1  # for debugging
    cfg.texturedPerVertexFittingCfg.imgSize = 1080
    cfg.texturedPerVertexFittingCfg.terminateLoss = 0.1
    cfg.texturedPerVertexFittingCfg.lpSmootherW = 1e-1
    cfg.texturedPerVertexFittingCfg.normalSmootherW = 0.0
    cfg.texturedPerVertexFittingCfg.numIterations = 500
    cfg.texturedPerVertexFittingCfg.useKeypoints = False
    cfg.texturedPerVertexFittingCfg.kpFixingWeight = 0
    cfg.texturedPerVertexFittingCfg.noiseLevel = 0.1
    cfg.texturedPerVertexFittingCfg.bodyJointOnly = True
    cfg.texturedPerVertexFittingCfg.jointRegularizerWeight = 1e-5
    cfg.texturedPerVertexFittingCfg.toSparseCornersFixingWeight = 1
    cfg.texturedPerVertexFittingCfg.bin_size = 256

    texturedFitting(inputs, cfg)





