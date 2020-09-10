from M04_TexturedFitting import texturedPoseFitting, texturedPerVertexFitting
from Utility import *
import tqdm, copy
import torch
from Config import RenderingCfg
import shutil
from M04_ObjConverter import converObjsInFolder

class Config:
    def __init__(s):
        s.texturedPoseFittingCfg = RenderingCfg()
        s.texturedPerVertexFittingCfg = RenderingCfg()

class InputBundle():
    def __init__(s):
        s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'
        s.smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
        s.cleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
        s.texturedMesh = r"..\Data\TextureMap2Color\SMPLWithSocks_tri.obj"
        s.compressedStorage = True

        s.dataFolder = None
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = None
        s.toSparseFittedFolder = None
        s.outputFolderAll = None
        s.outputFolderFinal = None
        s.initialFittingParamFile = None

def texturedFitting(inputs, frameNames, cfg=Config()):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    for iF in tqdm.tqdm(range(len(frameNames)), desc='Textured Fitting: '):
        frameName = frameNames[iF]
        inputsPoseFitting = copy.copy(inputs)
        inputsPoseFitting.imageFolder = join(inputs.dataFolder, 'Preprocessed',  frameName)
        inputsPoseFitting.sparsePointCloudFile = join(inputs.inOriginalObjFilesFolder,  'A'+frameName.zfill(8) + '.obj')
        inputsPoseFitting.KeypointsFile = join(inputs.outputFolderAll, 'Keypoints', frameName + '.obj')
        inputsPoseFitting.outputFolder = join(inputs.outputFolderAll, 'PoseFitting', frameName)
        inputsPoseFitting.initialFittingParamFile = join(inputs.dataFolder, 'ToSparse',  frameName, 'InterpolatedParams.npz')

        os.makedirs(inputsPoseFitting.outputFolder, exist_ok=True)
        texturedPoseFitting(inputsPoseFitting, cfg.texturedPoseFittingCfg, device)

    #     paramFiles = sortedGlob(join(inputsPoseFitting.outputFolder, 'FitParam', '*.npz'))
    #     paramFiles.sort()
    #     finalPoseFile = paramFiles[-1]
    #     inputsPerVertexFitting = copy.copy(inputsPoseFitting)
    #     inputsPerVertexFitting.imageFolder = join(inputs.dataFolder, 'Preprocessed',  frameName)
    #     inputsPerVertexFitting.compressedStorage = True
    #     inputsPerVertexFitting.initialFittingParamFile = finalPoseFile
    #     inputsPerVertexFitting.outputFolder = join(inputs.outputFolderAll, 'PerVertexFitting', frameName)
    #     os.makedirs(inputsPerVertexFitting.outputFolder, exist_ok=True)
    #
    #     # texturedPerVertexFitting(inputsPerVertexFitting, cfg.texturedPerVertexFittingCfg, device)
    #
    #     # copy final files
        finalMeshFolder = join(inputs.outputFolderFinal, 'Mesh')
        finalParamFolder = join(inputs.outputFolderFinal, 'Params')
    #
        os.makedirs(finalMeshFolder, exist_ok=True)
        os.makedirs(finalParamFolder, exist_ok=True)

        meshFile = sortedGlob(join(inputsPoseFitting.outputFolder, 'Mesh', '*.ply'))[-1]
        paramFile = sortedGlob(join(inputsPoseFitting.outputFolder, 'FitParam', '*.npz'))[-1]

        shutil.copy(meshFile, join(finalMeshFolder, 'A'+frameName + '.ply'))
        shutil.copy(paramFile, join(finalParamFolder, 'A'+frameName + '.npz'))
    converObjsInFolder(finalMeshFolder, join(finalMeshFolder, 'ObjWithUV'), ext='ply', convertToMM=True)

if __name__ == '__main__':
    inputs = InputBundle()
    frameNames = [
            # '08822', '08823'
            str(iFrame).zfill(5) for iFrame in range(8828, 8922)
        ]

    inputs.dataFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada'
    inputs.inOriginalObjFilesFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\TriangulationType1Only'
    inputs.toSparseFittedFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    # inputs.outputFolderAll = r'Z:\shareZ\2020_07_26_NewPipelineTestData\TexturedFitting'
    inputs.outputFolderAll = r'F:\WorkingCopy2\2020_08_07_TexturedFitting_MultiResoTest\TexturedFitting'
    inputs.outputFolderFinal = r'F:\WorkingCopy2\2020_08_07_TexturedFitting_MultiResoTest\Final'

    cfg = Config()
    cfg.texturedPoseFittingCfg.sigma = 1e-7
    cfg.texturedPoseFittingCfg.blurRange = 1e-7

    cfg.texturedPoseFittingCfg.plotStep = 100
    cfg.texturedPoseFittingCfg.numCams = 16
    # low learning rate for pose optimization
    cfg.texturedPoseFittingCfg.learningRate = 1e-4

    cfg.texturedPoseFittingCfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    # cfg.texturedPoseFittingCfg.imgSize = 1080
    # cfg.texturedPoseFittingCfg.batchSize = 2

    cfg.texturedPoseFittingCfg.imgSize = 540
    cfg.texturedPoseFittingCfg.batchSize = 4
    cfg.texturedPoseFittingCfg.faces_per_pixel = 5  # for debugging

    cfg.texturedPoseFittingCfg.terminateLoss = 0.1
    cfg.texturedPoseFittingCfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.texturedPoseFittingCfg.normalSmootherW = 0.0
    cfg.texturedPoseFittingCfg.numIterations = 2000
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
    cfg.texturedPoseFittingCfg.withSilhouette = False
    cfg.texturedPoseFittingCfg.extrinsicsOutsideCamera = True



    texturedFitting(inputs, frameNames, cfg)