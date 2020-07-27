import cv2
import json

import os, subprocess, copy

import numpy as np
from os.path import isfile, join

def visualizeCornerLocationToFile(inCorrFile, imgFiles, cornerId, outfolderName, neighborhoodCropSize = 100,
        codeSet = r'C:\Code\MyRepo\ChbCapture\04_Pipeline\GenerateModelSequenceMesh7\CID_no_meshVID.txt'):
    os.makedirs(outfolderName, exist_ok=True)

    corrData = json.load(open(inCorrFile))

    corrs = corrData["CorrPts"]
    numCams = len(corrs)

    for camId in range(numCams):
        cornerLocation = corrs[camId][cornerId]

        if cornerLocation[0] > 0 and cornerLocation[1] > 0:
            img = cv2.imread(imgFiles[camId], cv2.IMREAD_COLOR)

            imgPadded = cv2.copyMakeBorder(img, neighborhoodCropSize, neighborhoodCropSize, neighborhoodCropSize, neighborhoodCropSize,
                               cv2.BORDER_CONSTANT, value= [0, 0, 0])

            x = int(cornerLocation[0])
            y = int(cornerLocation[1])

            neighborhood = imgPadded[y:y+2*neighborhoodCropSize, x:x+2*neighborhoodCropSize]
            neighborhood[neighborhoodCropSize, neighborhoodCropSize] = [0, 0, 255]

            outFileFileName = r"%s\Cam%04d.png"%(outfolderName, camId)
            cv2.imwrite(outFileFileName,neighborhood)

def concatImgs(imgList, concatSize, imgScale):
    concatImg = None
    imgShape = imgList[0].shape
    for iR in range(concatSize[0]):
        for iC in range(concatSize[1]):
            iIm = concatSize[1] * iR + iC

            newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
            if imgScale != 1:
                newimg = cv2.resize(imgList[iIm], (int(newX), int(newY)), interpolation = cv2.INTER_NEAREST)
            else:
                newimg = imgList[iIm]
            if iC == 0:
                numpy_horizontal = copy.copy(newimg)
            else :
                numpy_horizontal = np.hstack((numpy_horizontal, newimg))
        if iR == 0:
            concatImg = copy.copy(numpy_horizontal)
        else:
            concatImg = np.vstack((concatImg, numpy_horizontal))

    return concatImg

def getCornerLoacation(inCorrFile, cornerId):
    corrData = json.load(open(inCorrFile))

    corrs = corrData["CorrPts"]
    numCams = len(corrs)
    cornerLocations = []

    for camId in range(numCams):
        cornerLocation = corrs[camId][cornerId]
        cornerLocations.append(cornerLocation)

    return cornerLocations,

def getCornerCropAndLocation(inCorrFile, imgFiles, cornerId, neighborhoodCropSize=50, cropCenters=None):
    corrData = json.load(open(inCorrFile))

    corrs = corrData["CorrPts"]
    numCams = len(corrs)
    corners = []
    crops = []
    cornersOriginal = []
    cropCentersOut = []
    for camId in range(numCams):
        cornerLocation = corrs[camId][cornerId]
        c = cornerLocation
        cornersOriginal.append(c)

        if cropCenters is None:

            cxi = int(c[0]) if c[0] - np.floor(c[0]) <= 0.5 else int(c[0]) + 1
            cyi = int(c[1]) if c[1] - np.floor(c[1]) <= 0.5 else int(c[1]) + 1
        else:
            if cropCenters[camId][0] > 0:
                cxi = cropCenters[camId][0]
                cyi = cropCenters[camId][1]
            else:
                cxi = int(c[0]) if c[0] - np.floor(c[0]) <= 0.5 else int(c[0]) + 1
                cyi = int(c[1]) if c[1] - np.floor(c[1]) <= 0.5 else int(c[1]) + 1

        corners.append([cornerLocation[0] - cxi + neighborhoodCropSize, cornerLocation[1] - cyi + neighborhoodCropSize])
        cropCentersOut.append([cxi, cyi])

        if cornerLocation[0] > 0 and cornerLocation[1] > 0:
            img = cv2.imread(imgFiles[camId], cv2.IMREAD_COLOR)
            # img = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)),
            #                           interpolation=cv2.INTER_NEAREST)

            imgPadded = cv2.copyMakeBorder(img, neighborhoodCropSize, neighborhoodCropSize, neighborhoodCropSize,
                                           neighborhoodCropSize, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            neighborhood = imgPadded[cyi:cyi + 2 * neighborhoodCropSize + 1, cxi:cxi + 2 * neighborhoodCropSize + 1, :]
            crops.append(neighborhood)
        else:
            crops.append(
                np.zeros((2 * neighborhoodCropSize + 1, 2 * neighborhoodCropSize + 1, 3), dtype=np.uint8))
    return np.array(crops), np.array(corners), np.array(cornersOriginal), cropCentersOut


def visualizeCornerLocation(inCorrFile, imgFiles, cornerId, stitchImg=True, maxCol = 4, neighborhoodCropSize=100, imgScale=3, putCoordText=True,
                                  codeSet=r'C:\Code\MyRepo\ChbCapture\04_Pipeline\GenerateModelSequenceMesh7\CID_no_meshVID.txt'):

    corrData = json.load(open(inCorrFile))

    corrs = corrData["CorrPts"]
    numCams = len(corrs)

    neighborhoodCropSizeScaled = int(neighborhoodCropSize * imgScale)

    cornermgs = []
    for camId in range(numCams):
        cornerLocation = corrs[camId][cornerId]

        if cornerLocation[0] > 0 and cornerLocation[1] > 0:
            img = cv2.imread(imgFiles[camId], cv2.IMREAD_COLOR)
            # img = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)),
            #                           interpolation=cv2.INTER_NEAREST)

            imgPadded = cv2.copyMakeBorder(img, neighborhoodCropSize, neighborhoodCropSize, neighborhoodCropSize,
                                           neighborhoodCropSize, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            c = cornerLocation
            cxi = int(c[0]) if c[0] - np.floor(c[0]) <= 0.5 else int(c[0]) + 1
            cyi = int(c[1]) if c[1] - np.floor(c[1]) <= 0.5 else int(c[1]) + 1

            neighborhood = imgPadded[cyi:cyi + 2 * neighborhoodCropSize + 1, cxi:cxi + 2 * neighborhoodCropSize + 1, :]
            neighborhood = cv2.resize(neighborhood, (int(neighborhoodCropSizeScaled * 2), int(neighborhoodCropSizeScaled * 2)),
                                      interpolation=cv2.INTER_NEAREST)

            cornerCenterX = int((neighborhoodCropSize + cornerLocation[0] - cxi - 0.5)*imgScale)
            cornerCenterY = int((neighborhoodCropSize + cornerLocation[1] - cyi - 0.5)*imgScale)

            neighborhood[cornerCenterX-5:cornerCenterX+5, cornerCenterY, :] = [0, 0, 255]
            neighborhood[cornerCenterX, cornerCenterY-5:cornerCenterY+5, :] = [0, 0, 255]
            if putCoordText:
                putTextOnImg(neighborhood, std)

            cornermgs.append(neighborhood)
        else:
            cornermgs.append(np.zeros((2*neighborhoodCropSizeScaled, 2*neighborhoodCropSizeScaled, 3), dtype=np.uint8))

    if stitchImg:
        numRows = int(numCams / maxCol)
        if numCams % maxCol:
            numRows = numRows +1

        stitched = concatImgs(cornermgs, (numRows, maxCol), 1)
        return stitched

    else:
        return cornermgs


def putTextOnImg(img, text, fontScale, org=[], fontThickness=2, orgType="Center"):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get boundary of this text
    textsize = cv2.getTextSize(text, font, fontScale, fontThickness)[0]

    # get coords based on boundary
    if len(org)!=2:
        org = (img.shape[1] / 2, img.shape[0] / 2)

    if orgType == "Center":
        textX = org[1] - (textsize[0]) / 2
        textY = org[0] - (textsize[1]) / 2
    elif orgType == "TopLeft":
        textX = org[1]
        textY = org[0]

    # add text centered on image
    cv2.putText(img, text, (int(textX), int(textY)), font, fontScale, (255, 255, 255), fontThickness)

    return img

def imgs2Vid(pathIn, vidOut, fps = 30, imgScale = 1, select = []):

    #pathIn = r'F:\WorkingCopy2\2019_04_16_8CamsCapture\VideoSequence\D\\'
    #pathOut = r'F:\WorkingCopy2\2019_04_16_8CamsCapture\VideoSequence\D.avi'
    #fps = 30

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])

    if len(select) != 2:
        iterRange = range(len(files))
    else:
        iterRange = range(select[0], select[1])

    for i in iterRange:
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
        newimg = cv2.resize(img, (int(newX), int(newY)))
        height, width, layers = newimg.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(newimg)
        #fourcc = cv2.VideoWriter_fourcc(*'')
    out = cv2.VideoWriter(vidOut, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
