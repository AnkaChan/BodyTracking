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
import copy

class Config:
    def __init__(s):
        s.kpReconCfg = M02_ReconstructionJointFromRealImagesMultiFolder.Config()
        s.kpReconCfg.doUndist = False
        s.kpReconCfg.convertToRGB = False
        s.saveDistRgb = False

        s.toSparseFittingCfg = M03_ToSparseFitting.Config()
        s.initWithLastFrameParam=True
        s.learningRateFollowingFrame = 0.005

        s.softConstraintWeight = 100

        s.converImg = True

def preprocessSelectedFrame(dataFolder, frameNames, camParamF, outFolder, cfg=Config()):
    # Select input Fodler
    camFolders = sorted(glob.glob(join(dataFolder, '*')))
    camNames = [os.path.basename(camFolder) for camFolder in camFolders]
    camParams = json.load(open(camParamF))['cam_params']
    camParams = [camParams[str(i)] for i in range(len(camParams))]

    outFolderUndist = join(outFolder, 'Preprocessed')
    outFolderKp = join(outFolder, 'Keypoints')
    os.makedirs(outFolderKp, exist_ok=True)

    if cfg.saveDistRgb:
        outFolderDist = join(outFolder, 'PreprocessedDist')
        os.makedirs(outFolderDist, exist_ok=True)

    for iF in tqdm.tqdm(range(len(frameNames)), desc='Preprocessing: '):
        frameName = frameNames[iF]
        inImgFilesCurFrame = [join(camFolders[iCam], camNames[iCam] + frameName + '.pgm') for iCam in range(len(camNames))]

        outFrameFolder = join(outFolderUndist, frameName)
        os.makedirs(outFrameFolder, exist_ok=True)

        rgbUndistFrameFiles = []
        for iCam, inImgF in enumerate(inImgFilesCurFrame):
            outImgFile = join(outFrameFolder, Path(inImgF).stem + '.png')
            if cfg.saveDistRgb:
                outFrameFolderDist = join(outFolderDist, frameName)
                os.makedirs(outFrameFolderDist, exist_ok=True)
                outImgFileDist = join(outFrameFolderDist, Path(inImgF).stem + '.png')
            else:
                outImgFileDist = None

            if cfg.converImg:
                M01_Preprocessing.preprocessImg(inImgF, outImgFile, camParams[iCam], outImgFileDist)

        rgbUndistFrameFiles = sortedGlob(join(outFrameFolder,  '*.png'))

        outKpFile = join(outFolderKp, frameName + '.obj')
        if cfg.kpReconCfg.drawResults:
            debugFolder = join(outFrameFolder, 'Debug')
        else:
            debugFolder = None
        M02_ReconstructionJointFromRealImagesMultiFolder.reconstructKeypoints2(rgbUndistFrameFiles, outKpFile, camParamF, cfg.kpReconCfg, debugFolder)

def toSparseFittingSelectedFrameV2(inputs, frameNames, cfg=Config()):
    json.dump(cfg.toSparseFittingCfg.__dict__, open(join(inputs.outFolderAll, 'Cfg.json'), 'w'))

    M03_ToSparseFitting.toSparseFittingNewRegressorV2(frameNames, inputs.inputKpFolder, inputs.deformedSparseMeshFolder, inputs.outFolderAll, inputs.skelDataFile, inputs.toSparsePCMat,
                        inputs.betaFile, inputs.personalShapeFile, inputs.SMPLSHNpzFile, initialPoseFile=inputs.fittingParamFile, cfg=cfg.toSparseFittingCfg)

def interpolateToSparseMeshSelectedFrame(inputs, frameNames, cfg=Config()):
    for iF in tqdm.tqdm(range(len(frameNames)), desc='Interpolating meshes: '):
        frameName = frameNames[iF]
        deformedSparseMeshFile = join(inputs.deformedSparseMeshFolder, 'A'+frameName.zfill(8) + '.obj')

        frameFittingFolder = join(inputs.outFolderAll, 'ToSparse', frameName)
        fitParamFile = join(frameFittingFolder, 'ToSparseFittingParams_withHH.npz')
        fittedMeshFile = join(frameFittingFolder, 'ToSparseMesh_withHH.obj')
        outInterpolatedMeshFile = join(frameFittingFolder, 'InterpolatedMesh.obj')
        outInterpolatedParamsFile = join(frameFittingFolder, 'InterpolatedParams.npz')

        M03_ToSparseFitting.getPersonalShapeFromInterpolation(fittedMeshFile, deformedSparseMeshFile, fitParamFile, outInterpolatedMeshFile, outInterpolatedParamsFile,
            inputs.skelDataFile, inputs.toSparsePCMat, laplacianMatFile=inputs.laplacianMatFile, smplshData=inputs.SMPLSHNpzFile,\
            handIndicesFile = r'HandIndices.json', HeadIndicesFile = 'HeadIndices.json', softConstraintWeight = cfg.softConstraintWeight,
            numRealCorners = 1487, fixHandAndHead = True, )

def figureOutHandAndHead(inputs, frameNames, cfg):
    for iF in tqdm.tqdm(range(len(frameNames)), desc='Figuring out hands and head: '):
        frameName = frameNames[iF]
        frameFittingFolder = join(inputs.outFolderAll, 'ToSparse', frameName)
        interpolatedParamsFile = join(frameFittingFolder, 'ToSparseFittingParams.npz')
        kpFile = join(inputs.inputKpFolder, frameName + '.obj')

        frameFittingFolder = join(inputs.outFolderAll, 'ToSparse', frameName)
        M03_ToSparseFitting.figureOutHandHeadPoses(interpolatedParamsFile, kpFile, inputs.SMPLSHNpzFile, frameFittingFolder, cfg=cfg.toSparseFittingCfg, personalShapeFile=inputs.personalShapeFile)

class InputBundle():
    def __init__(s, datasetName=r'Lada_12/12/2019'):
        if datasetName == r'Lada_12/12/2019':
            s.SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
            s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

            s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
            s.inputDensePointCloudFile = None
            s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
            s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
            s.betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'

        elif datasetName == r'Katey_01/01/2020_Remote':
            s.SMPLSHNpzFile = r'..\Data\BuildSmplsh_Female\Output\SmplshModel_f_noBun'
            s.skelDataFile = r'..\Data\KateyBodyModel\InitialRegistration\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'

            s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
            s.inputDensePointCloudFile = None
            s.toSparsePCMat = r'..\Data\KateyBodyModel\InterpolationMatrix.npy'
            s.personalShapeFile = r'..\Data\KateyBodyModel\\PersonalShape.npy'
            s.betaFile = r'..\Data\KateyBodyModel\Beta.npy'

        s.dataFolder = None
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = None
        s.outFolderAll = None
        s.laplacianMatFile = None
        s.fittingParamFile = None

if __name__ == '__main__':
    inputs = InputBundle()

    # inputs.dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.preprocessOutFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.deformedSparseMeshFolder = r''
    # inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\ObjFiles'
    # inputs.inputKpFolder = join(inputs.dataFolder, 'Keypoints')
    # inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inputs.outFolderAll = inputs.dataFolder
    # frameNames = [
    # #              '03067',
    #               # '03990',
    #               # '04735', '04917',
    #               # '06250',
    #               '06550',
    #               #  '06950'
    #               ]


    # inputs.dataFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada'
    # inputs.outFolderAll = inputs.dataFolder
    # inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LadaStand'
    # inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inputs.inputKpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'
    # inputs.outFolderAll = join(inputs.dataFolder, 'FitOnlyBody')
    #
    # frameNames = [str(iFrame).zfill(5) for iFrame in range(8564, 8564 + 50)]

    # Lada ground
    inputs.dataFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround'
    inputs.outFolderAll = inputs.dataFolder
    inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround\LadaGround'
    inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    inputs.inputKpFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround\Keypoints'
    inputs.outFolderAll = join(inputs.dataFolder, 'FitOnlyBody')

    frameNames = [str(iFrame).zfill(5) for iFrame in range(6141, 6141+100)]

    # inputs.dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.preprocessOutFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\ObjFiles'
    # inputs.inputKpFolder = join(inputs.dataFolder, 'Keypoints')
    # inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inputs.outFolderAll = join(inputs.dataFolder, 'FitOnlyBody')
    # frameNames = [
    #         '16755'
    #               ]

    cfg = Config()
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    cfg.toSparseFittingCfg.learnrate_first = 0.1
    cfg.toSparseFittingCfg.learnrate_following = 0.02
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.005
    cfg.toSparseFittingCfg.lrDecayStep = 200
    cfg.toSparseFittingCfg.lrDecayRate = 0.96
    cfg.toSparseFittingCfg.numIterFitting = 10000
    cfg.toSparseFittingCfg.terminateLoss = 1e-5
    cfg.toSparseFittingCfg.terminateLossStep = 1e-10
    cfg.toSparseFittingCfg.skeletonJointsToFix = [12, 15,]
    cfg.converImg = False
    cfg.kpReconCfg.openposeModelDir = r"C:\Code\Project\Openpose\models"


    # camFolders = sortedGlob(join(dataFolder, '*'))
    # imgFolders = sortedGlob(r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\*')
    #
    # for imgFolder in imgFolders[:-3]:
    #     imgFs = sortedGlob(join(imgFolder, '*.pgm'))
    #
    #     for imgF, camFolder in zip(imgFs, camFolders):
    #         shutil.copy(imgF, join(camFolder, os.path.basename(imgF)))
    # # preprocess
    # preprocessSelectedFrame(inputs.dataFolder, frameNames, inputs.camParamF, inputs.preprocessOutFolder, cfg)

    # to sparse fitting

    os.makedirs(inputs.outFolderAll, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    toSparseFittingSelectedFrameV2(inputs, frameNames, cfg)

    # intepolate to sparse mesh
    # interpolateToSparseMeshSelectedFrame(inputs, frameNames)