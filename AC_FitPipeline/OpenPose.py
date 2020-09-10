# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import  glob
import argparse
from pathlib import Path
import json
import numpy as np
from copy import  copy

# Import Openpose (Windows/Ubuntu/OSX)
#dir_path = os.path.dirname(os.path.realpath(__file__))
# opBinDir = r'C:\Code\Project\Openpose\bin'
# opReleaseDir = r'C:\Code\Project\Openpose\x64\Release'

opBinDir = r'Z:\Anka\OpenPose\bin'
opReleaseDir = r'Z:\Anka\OpenPose\x64\Release'
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


class OPWrapper2:
    def __init__(self):
        self.params = {"model_folder" : r"C:\Code\Project\Openpose\models",
                       'number_people_max' : 1,
                       "hand":True,
                       "hand_detector":2
                       }

        self.opWrapper = op.WrapperPython()
        self.datum = None

    def init(self):
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

        # Process Image
        self.datum = op.Datum()

def concatImgs(imgList, concatSize, imgScale):
    concatImg = None
    imgShape = imgList[0].shape
    for iR in range(concatSize[0]):
        for iC in range(concatSize[1]):
            iIm = concatSize[1] * iR + iC

            newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
            newimg = cv2.resize(imgList[iIm], (int(newX), int(newY)))

            if iC == 0:
                numpy_horizontal = copy(newimg)
            else :
                numpy_horizontal = np.hstack((numpy_horizontal, newimg))
        if iR == 0:
            concatImg = copy(numpy_horizontal)
        else:
            concatImg = np.vstack((concatImg, numpy_horizontal))

    return concatImg

if __name__ == "__main__":
    params = dict()
    params["model_folder"] = r"C:\Code\Project\Openpose\models"
    params['number_people_max'] = 1

    inFolders = [
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\A",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\B",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\C",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\D",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\E",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\F",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\G",
        r"F:\WorkingCopy2\2019_04_16_8CamsCapture\Converted\H",
    ]

    outFolderTotal = r'F:\WorkingCopy2\2019_04_16_8CamsCapture\Skeleton'

    fileRange = [1600,3200]

    opWrapper = op.WrapperPython()
    help(op)
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()


    for folder in inFolders:
        files = glob.glob(folder + r'\*.pgm')
        folderPath = Path(folder)
        folderName = folderPath.stem
        outFolder = outFolderTotal + '\\' + folderName
        os.makedirs(outFolder, exist_ok=True)
        for f in files[fileRange[0]:fileRange[1]]:
            imageToProcess = cv2.imread(f)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop([datum])
            filePath = Path(f)
            fileName = filePath.stem
            data = {}
            data['BodyKeypoints'] = datum.poseKeypoints.tolist()
            #print("Body keypoints: \n" + str(datum.poseKeypoints))
            #cv2.imshow("OpenPose 1.5.0 - Tutorial Python API", datum.cvOutputData)
            #cv2.imwrite("test.jpg", datum.cvOutputData)
            #cv2.waitKey(0)
            #print()
            outFile = outFolder + '\\' + fileName + '.json'
            with open(fileName, 'w') as outfile:
                json.dump(data, outfile)
