import OpenPose
import glob, sys, os
from pathlib import Path
import cv2
from os.path import join
from sys import platform

# Import Openpose (Windows/Ubuntu/OSX)
opBinDir = r'C:\Code\Project\Openpose\bin'
opReleaseDir = r'C:\Code\Project\Openpose\x64\Release'
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH']  = os.environ['PATH'] + ';' + opReleaseDir + ';' + opBinDir + ';'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


if __name__ == "__main__":
    # inFolder = r'F:\WorkingCopy2\2020_03_27_RenderTest\DataSMPLH\TPose'
    # inFolder = r'F:\WorkingCopy2\2020_03_27_RenderTest\DataSMPLH\Squat'
    # inFolder = r'F:\WorkingCopy2\2020_07_09_TestSimplify\ImageData'
    # inFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03052\toRGB'
    inFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\toRGB\Undist'
    extName = 'png'
    # showDetection = False
    showDetection = True

    inFiles = glob.glob(join(inFolder, r'*.' + extName))
    keypointsJsonDir = join(inFolder, 'KeypointsJson')

    os.makedirs(keypointsJsonDir, exist_ok=True)

    params = dict()
    # params["model_folder"] = "../../../models/"
    params["model_folder"] = r"C:\Code\Project\Openpose\models"
    params["face"] = True
    params["hand"] = True
    params["write_json"] = keypointsJsonDir

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    for imgF in inFiles:
        # f = inFiles[iCam][iP]
        datum = op.Datum()

        datum.name = Path(imgF).stem

        imageToProcess = cv2.imread(imgF)
        imgScale = 0.5
        imgShape = imageToProcess.shape
        newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
        newimg = cv2.resize(imageToProcess, (int(newX), int(newY)))
        datum.cvInputData = newimg
        opWrapper.emplaceAndPop([datum])
        filePath = Path(imgF)
        fileName = filePath.stem
        data = {}
        data['BodyKeypoints'] = datum.poseKeypoints.tolist()
        # data['BodyKeypoints'] = op.datum.poseKeypoints.tolist()

        print(datum.handKeypoints)
        if showDetection:
            resultImg = datum.cvOutputData
            cv2.imshow('resultImg', resultImg)
            cv2.waitKey(0)

