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
import sys

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

def detectKeypointsOnSelectedFrame(dataFolder, frameNames, camParamF, outFolder, cfg=Config(), openposeDir=r'C:\Code\Project\Openpose'):
    # Select input Fodler
    camFolders = sorted(glob.glob(join(dataFolder, '*')))
    camNames = [os.path.basename(camFolder) for camFolder in camFolders]
    camParams = json.load(open(camParamF))['cam_params']
    camParams = [camParams[str(i)] for i in range(len(camParams))]

    outFolderUndist = inputs.inputImgDataFolder
    outFolderKp = outFolder
    os.makedirs(outFolderKp, exist_ok=True)

    # Import Openpose (Windows/Ubuntu/OSX)
    opBinDir = join(openposeDir, 'bin')
    opReleaseDir = join(openposeDir, 'Release')
    # opBinDir = r'Z:\Anka\OpenPose\bin'
    # opReleaseDir = r'Z:\Anka\OpenPose\x64\Release'

    try:
        # Windows Import
        if sys.platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            # sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + opReleaseDir + ';' + opBinDir + ';'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    params = dict()
    params["model_folder"] = cfg.kpReconCfg.openposeModelDir
    params["face"] = cfg.kpReconCfg.detecHead
    params["hand"] = cfg.kpReconCfg.detectHand

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    for iF in tqdm.tqdm(range(len(frameNames)), desc='Preprocessing: '):
        frameName = frameNames[iF]
        inImgFilesCurFrame = [join(camFolders[iCam], camNames[iCam] + frameName + '.pgm') for iCam in range(len(camNames))]

        outFrameFolder = join(outFolderUndist, frameName)
        os.makedirs(outFrameFolder, exist_ok=True)

        rgbUndistFrameFiles = []
        # for iCam, inImgF in enumerate(inImgFilesCurFrame):
        #     outImgFile = join(outFrameFolder, Path(inImgF).stem + '.png')
        #     if cfg.saveDistRgb:
        #         outFrameFolderDist = join(outFolderDist, frameName)
        #         os.makedirs(outFrameFolderDist, exist_ok=True)
        #         outImgFileDist = join(outFrameFolderDist, Path(inImgF).stem + '.png')
        #     else:
        #         outImgFileDist = None
        #
        #     if cfg.converImg:
        #         M01_Preprocessing.preprocessImg(inImgF, outImgFile, camParams[iCam], outImgFileDist)

        rgbUndistFrameFiles = sortedGlob(join(outFrameFolder,  '*.png'))

        outKpFile = join(outFolderKp, frameName + '.json')
        if cfg.kpReconCfg.drawResults:
            debugFolder = join(outFrameFolder, 'Debug')
        else:
            debugFolder = None

        corrs = M02_ReconstructionJointFromRealImagesMultiFolder.detectKeyPoints(rgbUndistFrameFiles, outKpFile, camParamF, cfg.kpReconCfg, debugFolder, opWrapper=opWrapper)

        json.dumps({'Keypoints2D':corrs, 'ImageFiles':rgbUndistFrameFiles, 'cfg':cfg.kpReconCfg.__dict__})


class InputBundle():
    def __init__(s, datasetName=r'Lada_12/12/2019'):
        if datasetName == r'Lada_12/12/2019':
            s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
            s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

        elif datasetName == r'Katey_01/01/2020_Remote':
            s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'

        s.inputKpFolder = None

if __name__ == '__main__':
    inputs = InputBundle()

    # Lada ground
    inputs.inputImgDataFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Preprocessed'
    inputs.inputKpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'

    frameNames = [str(iFrame).zfill(5) for iFrame in range(8564, 8564+200)]

    cfg = Config()
    cfg.converImg = False
    cfg.kpReconCfg.openposeModelDir = r"C:\Code\Project\Openpose\models"
    cfg.kpReconCfg.drawResults = True
    detectKeypointsOnSelectedFrame(inputs.inputImgDataFolder, frameNames, inputs.camParamF, inputs.inputKpFolder, cfg)

    # to sparse fitting
