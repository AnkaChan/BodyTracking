# convert to
# Undist images
# Reconstruct keypoints
# Fit to sparse point cloud and keypoint
# Interpolate using sparse point cloud

import M01_Preprocessing
import M02_ReconstructionJointFromRealImagesMultiFolder
import M03_ToSparseFitting
from Utility import *
import json
import shutil
from pathlib import Path
import tqdm


class Config:
    def __init__(s):
        s.kpReconCfg = M02_ReconstructionJointFromRealImagesMultiFolder.Config()
        s.kpReconCfg.doUndist = False
        s.kpReconCfg.convertToRGB = False

        s.toSparseFittingCfg = M03_ToSparseFitting.Config()

def preprocessSelectedFrame(dataFolder, frameNames, camParamF, outFolder, cfg=Config()):
    # Select input Fodler
    camFolders = sorted(glob.glob(join(dataFolder, '*')))
    camNames = [os.path.basename(camFolder) for camFolder in camFolders]
    camParams = json.load(open(camParamF))['cam_params']
    camParams = [camParams[str(i)] for i in range(len(camParams))]

    outFolderUndist = join(outFolder, 'Preprocessed')
    outFolderKp = join(outFolder, 'Keypoints')
    os.makedirs(outFolderKp, exist_ok=True)

    for iF in tqdm.tqdm(range(len(frameNames)), desc='Preprocessing: '):
        frameName = frameNames[iF]
        inImgFilesCurFrame = [join(camFolders[iCam], camNames[iCam] + frameName + '.pgm') for iCam in range(len(camNames))]

        outFrameFolder = join(outFolderUndist, frameName)
        os.makedirs(outFrameFolder, exist_ok=True)

        rgbUndistFrameFiles = []
        for iCam, inImgF in enumerate(inImgFilesCurFrame):
            outImgFile = join(outFrameFolder, Path(inImgF).stem + '.png')
            M01_Preprocessing.preprocessImg(inImgF, outImgFile, camParams[iCam])
            rgbUndistFrameFiles.append(outImgFile)

        outKpFile = join(outFolderKp, frameName + '.obj')
        M02_ReconstructionJointFromRealImagesMultiFolder.reconstructKeypoints2(rgbUndistFrameFiles, outKpFile, camParamF, cfg.kpReconCfg, )

def toSparseFittingSelectedFrame(inputs, frameNames, cfg=Config()):
    for iF in tqdm.tqdm(range(len(frameNames)), desc='Fitting to Sparse: '):
        frameName = frameNames[iF]
        deformedSparseMeshFile = join(inputs.deformedSparseMeshFolder, 'A'+frameName.zfill(8) + '.obj')
        kpFile = join(inputs.inputKpFolder, frameName + '.obj')
        outputFolderForFrame = join(inputs.outFolderAll, frameName)
        os.makedirs(outputFolderForFrame, exist_ok=True)

        M03_ToSparseFitting.toSparseFittingNewRegressor(kpFile, deformedSparseMeshFile, outputFolderForFrame, inputs.skelDataFile, inputs.toSparsePCMat,
                                                        inputs.betaFile, inputs.personalShapeFile, inputs.SMPLSHNpzFile, cfg=cfg.toSparseFittingCfg)

def interpolateToSparseMeshSelectedFrame(inputs, frameNames, cfg=Config()):
    for iF in tqdm.tqdm(range(len(frameNames)), desc='Fitting to Sparse: '):
        frameName = frameNames[iF]
        deformedSparseMeshFile = join(inputs.deformedSparseMeshFolder, 'A'+frameName.zfill(8) + '.obj')

        frameFittingFolder = join(inputs.outFolderAll, frameName)
        fitParamFile = join(frameFittingFolder, 'ToSparseFittingParams.npz')
        fittedMeshFile = join(frameFittingFolder, 'ToSparseMesh.obj')
        outInterpolatedMeshFile = join(frameFittingFolder, 'InterpolatedMesh.obj')
        outInterpolatedParamsFile = join(frameFittingFolder, 'InterpolatedParams.npz')

        M03_ToSparseFitting.getPersonalShapeFromInterpolation(fittedMeshFile, deformedSparseMeshFile, fitParamFile, outInterpolatedMeshFile, outInterpolatedParamsFile,
            inputs.skelDataFile, inputs.toSparsePCMat, laplacianMatFile=None, smplshData=inputs.SMPLSHNpzFile,\
            handIndicesFile = r'HandIndices.json', HeadIndicesFile = 'HeadIndices.json', softConstraintWeight = 100,
            numRealCorners = 1487, fixHandAndHead = True, )


class InputBundle():
    def __init__(s):
        s.SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

        s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'

        s.dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
        s.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed'
        s.inputKpFolder = join(s.dataFolder, 'Keypoints')
        s.outFolderAll = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\ToSparse'


if __name__ == '__main__':
    # dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\Images'
    # preprocessOutFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # frameNames = ['03067',
    #               # '03990',
    #               '04735', '04917', '06250', '06550', '06950']

    dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\Images'
    preprocessOutFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    frameNames = ['03067',
                  # '03990',
                  '04735', '04917', '06250', '06550', '06950']

    cfg = Config()
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    cfg.toSparseFittingCfg.learnrate_ph = 0.05
    cfg.toSparseFittingCfg.lrDecayStep = 200
    cfg.toSparseFittingCfg.lrDecayRate = 0.96
    cfg.toSparseFittingCfg.numIterFitting = 8000
    cfg.toSparseFittingCfg.noBodyKeyJoint = True
    cfg.toSparseFittingCfg.betaRegularizerWeightToKP = 1000

    cfg.kpReconCfg.openposeModelDir = r"C:\Code\Project\Openpose\models"
    inputs = InputBundle()

    # camFolders = sortedGlob(join(dataFolder, '*'))
    # imgFolders = sortedGlob(r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\*')
    #
    # for imgFolder in imgFolders[:-3]:
    #     imgFs = sortedGlob(join(imgFolder, '*.pgm'))
    #
    #     for imgF, camFolder in zip(imgFs, camFolders):
    #         shutil.copy(imgF, join(camFolder, os.path.basename(imgF)))
    # preprocess
    # preprocessSelectedFrame(dataFolder, frameNames, camParamF, preprocessOutFolder, cfg)

    # to sparse fitting

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # toSparseFittingSelectedFrame(inputs, frameNames, cfg)

    # intepolate to sparse mesh
    interpolateToSparseMeshSelectedFrame(inputs, frameNames)