from M04_TexturedFitting import texturedPoseFitting, texturedPerVertexFitting
from Utility import *
import tqdm, copy
import torch
from S02_TexturedFittingForSelectedFrames import *

class InputBundle():
    def __init__(s):
        s.camParamF = r'F:\WorkingCopy2\2020_01_01_KateyCapture\CameraParameters2_k1k2k3p1p2\cam_params.json'
        s.smplshData = r'..\Data\BuildSmplsh_Female\Output\SmplshModel_f_noBun.npz'
        s.skelDataFile = r'..\Data\KateyBodyModel\InitialRegistration\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'..\Data\KateyBodyModel\InterpolationMatrix.npy'
        s.personalShapeFile = r'..\Data\KateyBodyModel\PersonalShape.npy'
        s.betaFile = r'..\Data\KateyBodyModel\beta.npy'
        s.smplshExampleMeshFile = r'..\SMPL_reimp\SMPLSH.obj'
        s.cleanPlateFolder = r'F:\WorkingCopy2\2020_09_10_CleanPlateKatey\Preprocessed\03280'
        s.texturedMesh = r"..\Data\KateyBodyModel\BodyMesh\Initial\Katey.obj"
        s.compressedStorage = True

        s.dataFolder = r'Z:\2020_01_01_KateyCapture\Converted'
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = join(s.dataFolder, 'Keypoints')
        s.toSparseFittedFolder = None
        s.outputFolderAll = None
        s.outputFolderFinal = None
        s.initialFittingParamFile = None

def texturedFittingV2(inputs, frameNames, cfg=Config()):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    for iF in tqdm.tqdm(range(len(frameNames)), desc='Textured Fitting: '):
        frameName = frameNames[iF]
        inputsPerVertexFitting = copy.copy(inputs)
        inputsPerVertexFitting.imageFolder = join(inputs.dataFolder, 'Preprocessed',  frameName)
        inputsPerVertexFitting.sparsePointCloudFile = join(inputs.inOriginalObjFilesFolder,  'A'+frameName.zfill(8) + '.obj')
        inputsPerVertexFitting.KeypointsFile = join(inputs.outputFolderAll, 'Keypoints', frameName + '.obj')
        inputsPerVertexFitting.outputFolder = join(inputs.outputFolderAll, 'PerVertexFitting', frameName)
        inputsPerVertexFitting.initialFittingParamFile = join(inputs.dataFolder, 'ToSparse',  frameName, 'InterpolatedParams.npz')

        os.makedirs(inputsPerVertexFitting.outputFolder, exist_ok=True)

        texturedPerVertexFitting(inputsPerVertexFitting, cfg.texturedPerVertexFittingCfg, device)

        # copy final files
        finalMeshFolder = join(inputs.outputFolderFinal, 'Mesh')
        finalParamFolder = join(inputs.outputFolderFinal, 'Params')

        os.makedirs(finalMeshFolder, exist_ok=True)
        os.makedirs(finalParamFolder, exist_ok=True)

        meshFile = sortedGlob(join(inputsPerVertexFitting.outputFolder, 'Mesh', '*.ply'))[-1]
        paramFile = sortedGlob(join(inputsPerVertexFitting.outputFolder, 'FitParam', '*.npz'))[-1]

        shutil.copy(meshFile, join(finalMeshFolder, 'A'+frameName + '.ply'))
        shutil.copy(paramFile, join(finalParamFolder, 'A'+frameName + '.npz'))
    converObjsInFolder(finalMeshFolder, join(finalMeshFolder, 'ObjWithUV'), ext='ply', convertToMM=True)

if __name__ == '__main__':

    inputs = InputBundle()
    frameNames = ['17126', '17438']

    inputs.dataFolder = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TextureCompletion'
    inputs.inOriginalObjFilesFolder = r'F:\WorkingCopy2\2020_01_13_FinalAnimations\Katey_NewPipeline\LongSequence\TriangulationType1Only'
    inputs.toSparseFittedFolder = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TextureCompletion\ToSparse'
    # inputs.outputFolderAll = r'Z:\shareZ\2020_07_26_NewPipelineTestData\TexturedFitting'
    inputs.outputFolderAll = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TextureCompletion\TexturedFitting'
    inputs.outputFolderFinal = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TextureCompletion\Final'

    cfg = Config()

    cfg.texturedPerVertexFittingCfg.sigma = 1e-7
    cfg.texturedPerVertexFittingCfg.blurRange = 1e-7

    cfg.texturedPerVertexFittingCfg.plotStep = 100
    # cfg.texturedPerVertexFittingCfg.plotStep = 20
    cfg.texturedPerVertexFittingCfg.numCams = 16
    # cfg.texturedPerVertexFittingCfg.learningRate = 1e-2 # for SGD
    cfg.texturedPerVertexFittingCfg.learningRate = 1e-4 # for Adam

    cfg.texturedPerVertexFittingCfg.faces_per_pixel = 1  # for debugging
    # cfg.texturedPerVertexFittingCfg.imgSize = 1080
    # cfg.texturedPerVertexFittingCfg.batchSize = 2

    cfg.texturedPerVertexFittingCfg.imgSize = 1080
    cfg.texturedPerVertexFittingCfg.batchSize = 4

    cfg.texturedPerVertexFittingCfg.terminateLoss = 0.1
    cfg.texturedPerVertexFittingCfg.lpSmootherW = 1e-2
    cfg.texturedPerVertexFittingCfg.normalSmootherW = 0.0
    cfg.texturedPerVertexFittingCfg.numIterations = 100
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
    cfg.texturedPerVertexFittingCfg.optimizerType = 'Adam'
    cfg.texturedPerVertexFittingCfg.terminateStep = 1e-7

    texturedFittingV2(inputs, frameNames, cfg)