import OpenPose
import glob, sys, os
from pathlib import Path
import cv2, subprocess
import json, tqdm
from os.path import join
from sys import platform
import numpy as np
# import MultiCameraReconstructFromRecog
import importlib.util
from SuitCapture import Camera
from SuitCapture import Triangulation
import pyvista as pv
from M01_Preprocessing import *

# pathToPyModule = r"E:\Projects\ChbCapture\MultiCameraReconstructOutputCorrespondences\Release\MultiCameraReconstructOutputCorrespondences.cp36-win_amd64.pyd"
# spec = importlib.util.spec_from_file_location("MultiCameraReconstructFromRecog", pathToPyModule)
# MultiCameraReconstructFromRecog = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(MultiCameraReconstructFromRecog)

from matplotlib import pyplot as plt

# print(MultiCameraReconstructFromRecog.helper())

class Config:
    def __init__(s):
        s.extName = 'png'
        s.confidenceThreshold = 0.30
        # number of to pick
        s.numMostConfidentToPick = 3
        # numMostConfidentToPick = -1
        s.reprojectErrThreshold = 10

        s.doRecon = True
        s.doUndist = True
        s.detectHand = True
        s.detecHead = True

        s.drawResults = False
        s.drawReconstructedKeypoints = False
        s.debugFolder = None

        s.keypointSkeletonParentTable = [
        # torso
        -1,  # 0
        0,   # 1
        1,   # 2
        2,   # 3
        3,   # 4
        1,   # 5
        5,   # 6
        6,   # 7
        1,   # 8
        8,   # 9
        9,   # 10
        10,  # 11
        8,   # 12
        12,  # 13
        13,  # 14
        0,   # 15
        0,   # 16
        15,  # 17
        16,  # 18
        14,  # 19
        14,  # 20
        14,  # 21
        11,  # 22
        11,  # 23
        11,  # 24
        # Left Hand
        7,   # 25
        25,  # 26
        26,  # 27
        27,  # 28
        28,  # 29
        25,  # 30
        30,  # 31
        31,  # 32
        32,  # 33
        25,  # 34
        34,  # 35
        35,  # 36
        36,  # 37
        25,  # 38
        38,  # 39
        39,  # 40
        40,  # 41
        25,  # 42
        42,  # 43
        43,  # 44
        44,  # 45
        # Right Hand
        4,   # 46
        46,  # 47
        47,  # 48
        48,  # 49
        49,  # 50
        46,  # 51
        51,  # 52
        52,  # 53
        53,  # 54
        46,  # 55
        55,  # 56
        56,  # 57
        57,  # 58
        46,  # 59
        59,  # 60
        60,  # 61
        61,  # 62
        46,  # 63
        63,  # 64
        64,  # 65
        65,  # 66
    ]
        s.openposeModelDir = r"Z:\Anka\OpenPose\models"

        s.convertToRGB = True

        s.rescale = False
        s.rescaleLvl = 0.5

class MultiCameraReconstructConfig:
    def __init__(self):
        self.maxReprojErr = 10.0
        self.reprojErrFile = ""
        self.doUndist = False
        self.outputExtName = "obj"
        self.RANSACRecon_iterAllPairs = True
        # After filtered out the outlier cameras, reconstruct the point again on all cameras
        # self.RANSACRecon_reconstructAllCams = True
        self.RANSACRecon_reconstructAllCams = False
        # 0 overall camera; 1 on selected pair; 2 overall camera exclude outlier
        self.RANSACRecon_PairReprojErrorScheme = 2
        self.RANSACRecon_OutlierFilderThres = 5.0
        self.RANSACRecon_HardRepjectionThres = 5.0

        self.minDetValForMin3DErr = 0.01
        self.verbose = True
        self.robustifierThres = 0.5


# Import Openpose (Windows/Ubuntu/OSX)
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


def filterOutUnconfidentRecog(keypoints, threshold):
    unconfidentRecog = np.where(keypoints[:, 2] < threshold)[0]
    if unconfidentRecog.shape[0]:
        keypoints[unconfidentRecog, :2] = [-1, -1]
    return  keypoints

def drawKeyPoints(outFile, img, keypoints, parentTable, keypointSize = 0.6, lineWidth = 0.1):
    fig, ax = plt.subplots()
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(im_rgb, vmin=0, vmax=255, interpolation='nearest')

    ax.plot(keypoints[:, 0], keypoints[:, 1], 'x', color='green', markeredgewidth=0.1, markersize=keypointSize)
    for iKP in range(1, keypoints[:67,0].shape[0]):
        parantKPId = parentTable[iKP]

        if keypoints[iKP, 0] > 0 and keypoints[parantKPId, 0] > 0:
            linePId = [parantKPId, iKP]
            ax.plot(keypoints[linePId, 0], keypoints[linePId, 1], linewidth=lineWidth)

    ax.axis('off')
    fig.savefig(outFile, dpi=2000, bbox_inches='tight', pad_inches=0)
    plt.close()

def project3DKeypoints(outFile, img, projMat, keypoints3D, parentTable, keypointSize = 0.6, lineWidth = 0.1):
    keyPts2D = []
    for iV in range(len(keypoints3D)):
        p3D = np.array(keypoints3D[iV])
        if np.all(p3D == [0,0,-1]):
            keyPts2D.append([-1,-1])
        else:
            p3DH = np.vstack([p3D.reshape(3, 1), 1])

            ptsP = projMat @ p3DH
            ptsP = ptsP / ptsP[2]

            # print(ptsP)
            keyPts2D.append(ptsP[:2, 0])
    keyPts2D = np.array(keyPts2D)
    drawKeyPoints(outFile, img, keyPts2D, parentTable, keypointSize, lineWidth)

def write_obj(file_name, verts):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def projectMesh(outFile, img, projMat, mesh):
    pts2D = []
    for iV in range(mesh.points.shape[0]):
        p3D = mesh.points[iV, :]
        p3DH = np.vstack([p3D.reshape(3, 1), 1])

        ptsP = projMat @ p3DH
        ptsP = ptsP / ptsP[2]

        # print(ptsP)
        pts2D.append(ptsP[:2, 0])

    pts2D = np.array(pts2D)
    # Draw projected pts on rendered image:
    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=255, interpolation='nearest', cmap=plt.get_cmap('gray'))

     # = join(outFolder, Path(imgFile).stem + '.png')

    ax.plot(pts2D[:, 0], pts2D[:, 1], '.', color='green', markeredgewidth=0.1, markersize=0.1)
    ax.axis('off')
    fig.savefig(outFile, dpi=2000, bbox_inches='tight', pad_inches=0)
    plt.close()

def projectReconstructedKeypointSkeleton(outFile, img, keypoints3D, parentTable):
    pass

def reconstructKeypoints(imgFolder, calibrationDataFile, cfg=Config()):
    params = dict()
    params["model_folder"] = cfg.openposeModelDir
    params["face"] = cfg.detecHead
    params["hand"] = cfg.detectHand

    if cfg.convertToRGB:
        inFolder = join(imgFolder, 'toRGB')

        examplePngFiles = glob.glob(join('ExampleFiles', '*.dng'))
        inGrayImgFiles = glob.glob(join(imgFolder, r'*.pgm'))
        inverseConvertMultiCams(inGrayImgFiles, inFolder, examplePngFiles)
    else:
        inFolder = imgFolder

    inFiles = glob.glob(join(inFolder, r'*.' + cfg.extName))
    camParams, camNames = Camera.loadCamParams(calibrationDataFile)

    camProjMats = []
    for iCam in range(len(camParams)):
        camParam = camParams[iCam]
        I, E = Camera.calibrationParamsToIEMats(camParam, True)

        projMat = I @ E
        # pts2D = Triangulation.projectPoints(mesh.points, projMat)
        # pts2Ds.append(pts2D)
        camProjMats.append(projMat)

    undistImageFolder = join(inFolder, 'Undist')
    os.makedirs(undistImageFolder, exist_ok=True)

    if cfg.doUndist:
        for imgF, camParam in zip(inFiles, camParams):
            # f = inFiles[iCam][iP]
            datum = op.Datum()
            fx = camParam['fx']
            fy = camParam['fy']
            cx = camParam['cx']
            cy = camParam['cy']
            intrinsic_mtx = np.array([
                [fx, 0.0, cx, ],
                [0.0, fy, cy],
                [0.0, 0.0, 1],
            ])

            undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
                                           camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])

            img = cv2.imread(imgF)
            imgUndist = cv2.undistort(img, intrinsic_mtx, undistortParameter)
            #
            # if removeBayerPattern:
            #     imgUndist = imgUndist[::2, ::2]

            outUndistImgFile = join(undistImageFolder, Path(imgF).stem + '.' + cfg.extName)
            cv2.imwrite(outUndistImgFile, imgUndist)

    if cfg.doUndist:
        undistImgFiles = glob.glob(join(undistImageFolder, '*.' + cfg.extName))
    else:
        undistImgFiles = glob.glob(join(inFolder, '*.' + cfg.extName))
    undistImgFiles.sort()

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    corrs = []
    folderKeypoints = join(inFolder, 'Keypoints')
    folderKeypointsWithSkel = join(inFolder, 'KeypointsWithSkeleton')
    # folderProjectedMesh = join(inFolder, 'Projected3DMesh')
    folderProjectedKP = join(inFolder, 'Projected3DKeypoints')
    os.makedirs(folderKeypoints, exist_ok=True)
    os.makedirs(folderKeypointsWithSkel, exist_ok=True)
    # os.makedirs(folderProjectedMesh, exist_ok=True)
    os.makedirs(folderProjectedKP, exist_ok=True)

    imgs = []
    datum = op.Datum()

    for iCam, imgF in enumerate(undistImgFiles):

        imageToProcess = cv2.imread(imgF)
        imgs.append(imageToProcess)
        # imgScale = 0.5
        # imgShape = imageToProcess.shape
        # newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
        # imageToProcess = cv2.resize(imageToProcess, (int(newX), int(newY)))
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        filePath = Path(imgF)
        fileName = filePath.stem
        data = {}
        data['BodyKeypoints'] = datum.poseKeypoints.tolist()
        # data['BodyKeypoints'] = op.datum.poseKeypoints.tolist()
        # print(datum.handKeypoints)

        # resultImg = datum.cvOutputData
        # cv2.imshow('resultImg', resultImg)
        # cv2.waitKey(0)
        # cv2.imwrite(outResultOImgFile, resultImg)

        keypoints = []

        if len(datum.poseKeypoints.shape) == 3:
            bodyKeyPoints = np.copy(datum.poseKeypoints)[0, :, :]
        else:
            bodyKeyPoints = np.copy(datum.poseKeypoints)

        # print(bodyKeyPoints)
        bodyKeyPoints = filterOutUnconfidentRecog(bodyKeyPoints, cfg.confidenceThreshold)
        # print(bodyKeyPoints)

        handKeyPointsLeft = np.copy(datum.handKeypoints[0])[0, :, :]
        handKeyPointsLeft = filterOutUnconfidentRecog(handKeyPointsLeft, cfg.confidenceThreshold)

        handKeyPointsRight = np.copy(datum.handKeypoints[1])[0, :, :]
        handKeyPointsRight = filterOutUnconfidentRecog(handKeyPointsRight, cfg.confidenceThreshold)

        keypoints = np.vstack([bodyKeyPoints, handKeyPointsLeft, handKeyPointsRight])

        corrs.append(keypoints.tolist())

        if cfg.drawResults:
            # outResultOImgFile = join(folderKeypoints, Path(imgF).stem + '.pdf')
            outResultOImgFile = join(folderKeypoints, Path(imgF).stem + '.png')
            drawKeyPoints(outResultOImgFile, imageToProcess, keypoints, cfg.keypointSkeletonParentTable)

            # outResultOImgWithSkelFile = join(folderKeypointsWithSkel, Path(imgF).stem + '.png')
            # drawKeyPoints(outResultOImgWithSkelFile, datum.cvOutputData, keypoints)

            # I, dist, E = Camera.loadCalibrationXML(calibXMLFiles[iCam])
            #
            # I4 = np.eye(4)
            # I4[:3,:3] = I
            # outProjectedMesh = join(folderProjectedMesh, Path(imgF).stem + '.png')
            # projectMesh(outProjectedMesh, imageToProcess, I4 @ E, mesh)

    reconFolder = join(inFolder, 'Reconstruction')
    os.makedirs(reconFolder, exist_ok=True)
    keypointsCorrsFile = join(reconFolder, 'KeypointsCorrs.json')

    json.dump(corrs, open(keypointsCorrsFile, 'w'))

    keypointsReconFile = join(reconFolder, 'KeypointsReconstruction.json')
    # reconConfig = MultiCameraReconstructConfig()
    # MultiCameraReconstructFromRecog.reconstructFromCorrsPy(corrs, calibXMLFiles, keypointsReconFile, reconConfig)
    # MultiCameraReconstructFromRecog.outputCorrespondencesPy(corrs, calibXMLFiles, keypointsReconFile, reconConfig)

    # triangulate using python code
    triangulations = []
    for i in range(len(corrs[0])):
        camPts = []
        selectedCamProjMats = []
        confidence = []
        for iCam in range(len(camProjMats)):
            if corrs[iCam][i][0] != -1:
                camPts.append(corrs[iCam][i])
                confidence.append(corrs[iCam][i][2])
                selectedCamProjMats.append(camProjMats[iCam])

        if cfg.numMostConfidentToPick != -1:
            sortedId = np.argsort(-np.array(confidence))
            # print(sortedId)
            selecedIds = sortedId[:cfg.numMostConfidentToPick]
            camPts = [camPts[i] for i in selecedIds]
            selectedCamProjMats = [selectedCamProjMats[i] for i in selecedIds]

        try:
            keyPoint3D, errs = Triangulation.mulCamsDLT(camPts, selectedCamProjMats)

            # print(errs.shape[0], errs)

            if np.mean(errs) < cfg.reprojectErrThreshold:
                triangulations.append(keyPoint3D)
            else:
                triangulations.append([0, 0, -1])
        except:
            triangulations.append([0, 0, -1])
    if cfg.drawReconstructedKeypoints:
        for iCam, (img, imgF) in enumerate(zip(imgs, undistImgFiles)):
            fx = camParam['fx']
            fy = camParam['fy']
            cx = camParam['cx']
            cy = camParam['cy']
            intrinsic_mtx = np.array([
                [fx, 0.0, cx, ],
                [0.0, fy, cy],
                [0.0, 0.0, 1],
            ])

            undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
                                           camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])

            img = imgs[iCam]
            imgUndist = cv2.undistort(img, intrinsic_mtx, undistortParameter)

            outResultOImgFile = join(folderProjectedKP, Path(imgF).stem + '.png')
            project3DKeypoints(outResultOImgFile, imgUndist, camProjMats[iCam], triangulations, cfg.keypointSkeletonParentTable)

    # reconstructionData = json.load(open(keypointsReconFile))
    # triangulations = reconstructionData['Triangulation']

    outTriangulationObjFile = join(reconFolder, "PointCloud.obj")
    write_obj(outTriangulationObjFile, triangulations)

def getFirstPersonKp(kpData):
    if len(kpData.shape) == 3:
        bodyKeyPoints = np.copy(kpData)[0, :, :]
    else:
        bodyKeyPoints = np.copy(kpData)

    return bodyKeyPoints

def reconstructKeypoints2(imgFiles, outTriangulationObjFile, calibrationDataFile, cfg=Config(), debugFolder=None):
    params = dict()
    params["model_folder"] = cfg.openposeModelDir
    params["face"] = cfg.detecHead
    params["hand"] = cfg.detectHand

    numBodypts = 25
    numHandpts = 21
    numFacepts = 70

    if cfg.convertToRGB:
        examplePngFiles = glob.glob(join('ExampleFiles', '*.dng'))
        imgs = inverseConvertMultiCams(imgFiles, None, examplePngFiles, writeFiles=False)
    else:
        imgs = [cv2.imread(imgF) for imgF in imgFiles]

    camParams, camNames = Camera.loadCamParams(calibrationDataFile)

    camProjMats = []
    for iCam in range(len(camParams)):
        camParam = camParams[iCam]
        I, E = Camera.calibrationParamsToIEMats(camParam, True)

        projMat = I @ E
        # pts2D = Triangulation.projectPoints(mesh.points, projMat)
        # pts2Ds.append(pts2D)
        camProjMats.append(projMat)

    # if cfg.doUndist:
    #     for imgF, camParam in zip(inFiles, camParams):
    #         # f = inFiles[iCam][iP]
    #         datum = op.Datum()
    #         fx = camParam['fx']
    #         fy = camParam['fy']
    #         cx = camParam['cx']
    #         cy = camParam['cy']
    #         intrinsic_mtx = np.array([
    #             [fx, 0.0, cx, ],
    #             [0.0, fy, cy],
    #             [0.0, 0.0, 1],
    #         ])
    #
    #         undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
    #                                        camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])
    #
    #         img = cv2.imread(imgF)
    #         imgUndist = cv2.undistort(img, intrinsic_mtx, undistortParameter)
    #         #
    #         # if removeBayerPattern:
    #         #     imgUndist = imgUndist[::2, ::2]
    #
    #         outUndistImgFile = join(undistImageFolder, Path(imgF).stem + '.' + cfg.extName)
    #         cv2.imwrite(outUndistImgFile, imgUndist)
    #
    # if cfg.doUndist:
    #     undistImgFiles = glob.glob(join(undistImageFolder, '*.' + cfg.extName))
    # else:
    #     undistImgFiles = glob.glob(join(inFolder, '*.' + cfg.extName))
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    corrs = []

    datum = op.Datum()

    for iCam in range(len(imgs)):
        imageToProcess = imgs[iCam]
        if cfg.convertToRGB:
            imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_RGB2BGR)
        if cfg.rescale:
            imgScale = cfg.rescaleLvl
            imgShape = imageToProcess.shape
            newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
            imageToProcess = cv2.resize(imageToProcess, (int(newX), int(newY)))
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        data = {}
        data['BodyKeypoints'] = datum.poseKeypoints.tolist()

        # resultImg = datum.cvOutputData
        # cv2.imshow('resultImg', resultImg)
        # cv2.waitKey(0)
        # cv2.imwrite(outResultOImgFile, resultImg)

        bodyKeyPoints = getFirstPersonKp(datum.poseKeypoints)
        if len(bodyKeyPoints.shape) and bodyKeyPoints.shape[0] == numBodypts:
            bodyKeyPoints = filterOutUnconfidentRecog(bodyKeyPoints, cfg.confidenceThreshold)
        else:
            bodyKeyPoints = np.zeros((numBodypts, 3))
        # print(bodyKeyPoints)

        handKeyPointsLeft = getFirstPersonKp(datum.handKeypoints[0])
        if len(handKeyPointsLeft.shape) and handKeyPointsLeft.shape[0] == numHandpts:
            handKeyPointsLeft = filterOutUnconfidentRecog(handKeyPointsLeft, cfg.confidenceThreshold)
        else:
            handKeyPointsLeft = np.zeros((numHandpts, 3))

        handKeyPointsRight = getFirstPersonKp(datum.handKeypoints[1])
        if len(handKeyPointsRight.shape) and handKeyPointsRight.shape[0] == numHandpts:
            handKeyPointsRight = filterOutUnconfidentRecog(handKeyPointsRight, cfg.confidenceThreshold)
        else:
            handKeyPointsRight = np.zeros((numHandpts, 3))

        keypoints = np.vstack([bodyKeyPoints, handKeyPointsLeft, handKeyPointsRight])

        # reconstruct facial keypoints
        if cfg.detecHead:
            faceKeyPoints = getFirstPersonKp(datum.faceKeypoints)
            if len(faceKeyPoints.shape) and faceKeyPoints.shape[0] == numFacepts:
                faceKeyPoints = filterOutUnconfidentRecog(faceKeyPoints, cfg.confidenceThreshold)
            else:
                faceKeyPoints = np.zeros((numBodypts, 3))

            keypoints = np.vstack([keypoints,faceKeyPoints])

        if cfg.rescale:
            goodKpIds = np.where(keypoints[:,0]>=0)[0]
            keypoints[goodKpIds,:2] = keypoints[goodKpIds,:2]/cfg.rescaleLvl

        camParam = camParams[iCam]
        fx = camParam['fx']
        fy = camParam['fy']
        cx = camParam['cx']
        cy = camParam['cy']
        intrinsic_mtx = np.array([
                [fx, 0.0, cx, ],
                [0.0, fy, cy],
                [0.0, 0.0, 1],
        ], dtype=np.float32)
        undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
                                           camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']], dtype=np.float32)
        keypointsPadded = keypoints[:, None, :2]
        if cfg.doUndist:
            keypointsPaddedUndist = cv2.undistortPoints(keypointsPadded, intrinsic_mtx, undistortParameter, P=intrinsic_mtx)
            keypointsPaddedUndist = np.squeeze(keypointsPaddedUndist)
        else:
            keypointsPaddedUndist = np.squeeze(np.squeeze(keypointsPadded))

        keypointsPaddedUndist[np.where(keypoints[:, 0] == -1)[0], :] = [-1, -1]

        corrs.append(np.hstack([keypointsPaddedUndist, keypoints[:,2:3]]).tolist())

        if cfg.drawResults:
            detectionDrawFolder = join(debugFolder, 'Detection')
            os.makedirs(detectionDrawFolder, exist_ok=True)
            # outResultOImgFile = join(folderKeypoints, Path(imgF).stem + '.pdf')
            outResultOImgFile = join(detectionDrawFolder, Path(imgFiles[iCam]).stem + '.png')
            drawKeyPoints(outResultOImgFile, imageToProcess, keypoints * cfg.rescaleLvl, cfg.keypointSkeletonParentTable)

            # outResultOImgWithSkelFile = join(folderKeypointsWithSkel, Path(imgF).stem + '.png')
            # drawKeyPoints(outResultOImgWithSkelFile, datum.cvOutputData, keypoints)

            # I, dist, E = Camera.loadCalibrationXML(calibXMLFiles[iCam])
            #
            # I4 = np.eye(4)
            # I4[:3,:3] = I
            # outProjectedMesh = join(folderProjectedMesh, Path(imgF).stem + '.png')
            # projectMesh(outProjectedMesh, imageToProcess, I4 @ E, mesh)

    # keypointsCorrsFile = join(reconFolder, 'KeypointsCorrs.json')
    # json.dump(corrs, open(keypointsCorrsFile, 'w'))
    # keypointsReconFile = join(reconFolder, 'KeypointsReconstruction.json')

    # triangulate using python code
    triangulations = []
    for i in range(len(corrs[0])):
        camPts = []
        selectedCamProjMats = []
        confidence = []
        for iCam in range(len(camProjMats)):
            if corrs[iCam][i][0] != -1:
                camPts.append(corrs[iCam][i])
                confidence.append(corrs[iCam][i][2])
                selectedCamProjMats.append(camProjMats[iCam])

        if cfg.numMostConfidentToPick != -1:
            sortedId = np.argsort(-np.array(confidence))
            # print(sortedId)
            selecedIds = sortedId[:cfg.numMostConfidentToPick]
            camPts = [camPts[i] for i in selecedIds]
            selectedCamProjMats = [selectedCamProjMats[i] for i in selecedIds]

        try:
            keyPoint3D, errs = Triangulation.mulCamsDLT(camPts, selectedCamProjMats)

            # print(errs.shape[0], errs)

            if np.mean(errs) < cfg.reprojectErrThreshold:
                triangulations.append(keyPoint3D)
            else:
                triangulations.append([0, 0, -1])
        except:
            triangulations.append([0, 0, -1])

    if cfg.drawReconstructedKeypoints:
        for iCam, (img, imgF) in enumerate(zip(imgs, imgFiles)):
            outResultOImgFile = join(debugFolder, Path(imgF).stem + '.png')
            project3DKeypoints(outResultOImgFile, img, camProjMats[iCam], triangulations, cfg.keypointSkeletonParentTable)

    # reconstructionData = json.load(open(keypointsReconFile))
    # triangulations = reconstructionData['Triangulation']

    # outTriangulationObjFile = join(reconFolder, "PointCloud.obj")
    write_obj(outTriangulationObjFile, triangulations)

if __name__ == '__main__':
    inParentFolder = r'Z:\shareZ\2019_12_13_Lada_Capture\Converted'
    outFolder = r'Z:\shareZ\2019_12_13_Lada_Capture\KeyPoints'
    cfg = Config()
    cfg.rescale = True
    # cfg.drawReconstructedKeypoints = True
    # cfg.rescale = False
    # reconInterval = [8853, 10853]
    # reconInterval = [8724, 8734]
    reconInterval = [9109, 10853]

    camFolders = glob.glob(join(inParentFolder, '*'))
    camFolders.sort()
    allImgs = [glob.glob(join(camFolder, '*.pgm')) for camFolder in camFolders]
    for iCam in range(len(allImgs)):
        allImgs[iCam].sort()

    debugFolder = join(outFolder, 'Debug')
    os.makedirs(debugFolder, exist_ok=True)

    camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
    os.makedirs(outFolder, exist_ok=True)
    for iFrame in tqdm.tqdm(range(reconInterval[0], reconInterval[1])):
        imgFiles = [camFiles[iFrame] for camFiles in allImgs]
        outFile = join(outFolder, str(iFrame).zfill(5)+'.obj')
        debugFolderFrame = join(debugFolder, str(iFrame).zfill(5))
        os.makedirs(debugFolderFrame, exist_ok=True)
        reconstructKeypoints2(imgFiles, outFile, camParamF, cfg, debugFolder=debugFolderFrame)


# if __name__ == '__main__':
#     inParentFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Images'
#     # extName = 'jpg'\
#     extName = 'png'
#
#     keypointSkeletonParentTable = [
#         # torso
#         -1,  # 0
#         0,   # 1
#         1,   # 2
#         2,   # 3
#         3,   # 4
#         1,   # 5
#         5,   # 6
#         6,   # 7
#         1,   # 8
#         8,   # 9
#         9,   # 10
#         10,  # 11
#         8,   # 12
#         12,  # 13
#         13,  # 14
#         0,   # 15
#         0,   # 16
#         15,  # 17
#         16,  # 18
#         14,  # 19
#         14,  # 20
#         14,  # 21
#         11,  # 22
#         11,  # 23
#         11,  # 24
#         # Left Hand
#         7,   # 25
#         25,  # 26
#         26,  # 27
#         27,  # 28
#         28,  # 29
#         25,  # 30
#         30,  # 31
#         31,  # 32
#         32,  # 33
#         25,  # 34
#         34,  # 35
#         35,  # 36
#         36,  # 37
#         25,  # 38
#         38,  # 39
#         39,  # 40
#         40,  # 41
#         25,  # 42
#         42,  # 43
#         43,  # 44
#         44,  # 45
#         # Right Hand
#         4,   # 46
#         46,  # 47
#         47,  # 48
#         48,  # 49
#         49,  # 50
#         46,  # 51
#         51,  # 52
#         52,  # 53
#         53,  # 54
#         46,  # 55
#         55,  # 56
#         56,  # 57
#         57,  # 58
#         46,  # 59
#         59,  # 60
#         60,  # 61
#         61,  # 62
#         46,  # 63
#         63,  # 64
#         64,  # 65
#         65,  # 66
#     ]
#
#     calibDataFile = r'CameraParameters_Lada\cam_params.json'
#     params = dict()
#     params["model_folder"] = r"C:\Code\Project\Openpose\models"
#     params["face"] = True
#     params["hand"] = True
#
#     threshold = 0.30
#     # number of to pick
#     numMostConfidentToPick = 3
#     # numMostConfidentToPick = -1
#     reprojectErrThreshold = 10
#
#     doRecon = True
#     # doRecon = False
#
#     doUndist = True
#
#     # drawResults = True
#     drawResults = False
#
#     # drawReconstructedKeypoints = True
#     drawReconstructedKeypoints = False
#
#     # removeBayerPattern = True
#     removeBayerPattern = False
#
#
#     inDataFolders = glob.glob(join(inParentFolder, '*'))
#
#     for inFolder in tqdm.tqdm(inDataFolders):
#         inFolder = join(inFolder, 'toRGB')
#         inFiles = glob.glob(join(inFolder, r'*.' + extName))
#         camParams, camNames = Camera.loadCamParams(calibDataFile)
#
#         camProjMats = []
#         for iCam in range(len(camParams)):
#             camParam = camParams[iCam]
#             I, E = Camera.calibrationParamsToIEMats(camParam, True)
#
#             projMat = I @ E
#             # pts2D = Triangulation.projectPoints(mesh.points, projMat)
#             # pts2Ds.append(pts2D)
#             camProjMats.append(projMat)
#
#         undistImageFolder = join(inFolder, 'Undist')
#         os.makedirs(undistImageFolder, exist_ok=True)
#
#         if doUndist:
#             for imgF, camParam in zip(inFiles, camParams):
#                 # f = inFiles[iCam][iP]
#                 datum = op.Datum()
#                 fx = camParam['fx']
#                 fy = camParam['fy']
#                 cx = camParam['cx']
#                 cy = camParam['cy']
#                 intrinsic_mtx = np.array([
#                     [fx, 0.0, cx, ],
#                     [0.0, fy, cy],
#                     [0.0, 0.0, 1],
#                 ])
#
#                 undistortParameter = np.array([camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
#                                                camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])
#
#                 img = cv2.imread(imgF)
#                 imgUndist = cv2.undistort(img, intrinsic_mtx, undistortParameter)
#
#                 if removeBayerPattern:
#                     imgUndist = imgUndist[::2, ::2]
#
#                 outUndistImgFile = join(undistImageFolder, Path(imgF).stem + '.' + extName)
#                 cv2.imwrite(outUndistImgFile, imgUndist)
#
#         if doUndist:
#             undistImgFiles = glob.glob(join(undistImageFolder, '*.' + extName))
#         else:
#             undistImgFiles = glob.glob(join(inFolder, '*.' + extName))
#         undistImgFiles.sort()
#
#         opWrapper = op.WrapperPython()
#         opWrapper.configure(params)
#         opWrapper.start()
#
#         corrs = []
#         folderKeypoints = join(inFolder, 'Keypoints')
#         folderKeypointsWithSkel = join(inFolder, 'KeypointsWithSkeleton')
#         # folderProjectedMesh = join(inFolder, 'Projected3DMesh')
#         folderProjectedKP = join(inFolder, 'Projected3DKeypoints')
#         os.makedirs(folderKeypoints, exist_ok=True)
#         os.makedirs(folderKeypointsWithSkel, exist_ok=True)
#         # os.makedirs(folderProjectedMesh, exist_ok=True)
#         os.makedirs(folderProjectedKP, exist_ok=True)
#
#         imgs = []
#
#         for iCam, imgF in enumerate( undistImgFiles):
#             datum = op.Datum()
#
#             imageToProcess = cv2.imread(imgF)
#             imgs.append(imageToProcess)
#             # imgScale = 0.5
#             # imgShape = imageToProcess.shape
#             # newX, newY = imgShape[1] * imgScale, imgShape[0] * imgScale
#             # imageToProcess = cv2.resize(imageToProcess, (int(newX), int(newY)))
#             datum.cvInputData = imageToProcess
#             opWrapper.emplaceAndPop([datum])
#             filePath = Path(imgF)
#             fileName = filePath.stem
#             data = {}
#             data['BodyKeypoints'] = datum.poseKeypoints.tolist()
#             # data['BodyKeypoints'] = op.datum.poseKeypoints.tolist()
#             # print(datum.handKeypoints)
#
#             resultImg = datum.cvOutputData
#             # cv2.imshow('resultImg', resultImg)
#             # cv2.waitKey(0)
#             # cv2.imwrite(outResultOImgFile, resultImg)
#
#             keypoints = []
#
#             if len(datum.poseKeypoints.shape) == 3:
#                 bodyKeyPoints = np.copy(datum.poseKeypoints)[0, :, :]
#             else:
#                 bodyKeyPoints = np.copy(datum.poseKeypoints)
#
#             # print(bodyKeyPoints)
#             bodyKeyPoints = filterOutUnconfidentRecog(bodyKeyPoints, threshold)
#             # print(bodyKeyPoints)
#
#             handKeyPointsLeft = np.copy(datum.handKeypoints[0])[0, :, :]
#             handKeyPointsLeft = filterOutUnconfidentRecog(handKeyPointsLeft, threshold)
#
#             handKeyPointsRight = np.copy(datum.handKeypoints[1])[0, :, :]
#             handKeyPointsRight = filterOutUnconfidentRecog(handKeyPointsRight, threshold)
#
#             keypoints = np.vstack([bodyKeyPoints, handKeyPointsLeft, handKeyPointsRight])
#
#             corrs.append(keypoints.tolist())
#
#             if drawResults:
#                 # outResultOImgFile = join(folderKeypoints, Path(imgF).stem + '.pdf')
#                 outResultOImgFile = join(folderKeypoints, Path(imgF).stem + '.png' )
#                 drawKeyPoints(outResultOImgFile, imageToProcess, keypoints, keypointSkeletonParentTable)
#
#                 # outResultOImgWithSkelFile = join(folderKeypointsWithSkel, Path(imgF).stem + '.png')
#                 # drawKeyPoints(outResultOImgWithSkelFile, datum.cvOutputData, keypoints)
#
#                 # I, dist, E = Camera.loadCalibrationXML(calibXMLFiles[iCam])
#                 #
#                 # I4 = np.eye(4)
#                 # I4[:3,:3] = I
#                 # outProjectedMesh = join(folderProjectedMesh, Path(imgF).stem + '.png')
#                 # projectMesh(outProjectedMesh, imageToProcess, I4 @ E, mesh)
#
#         reconFolder = join(inFolder, 'Reconstruction')
#         os.makedirs(reconFolder, exist_ok=True)
#         keypointsCorrsFile = join(reconFolder, 'KeypointsCorrs.json')
#
#         json.dump(corrs, open(keypointsCorrsFile, 'w'))
#
#         keypointsReconFile = join(reconFolder, 'KeypointsReconstruction.json')
#         # reconConfig = MultiCameraReconstructConfig()
#         # MultiCameraReconstructFromRecog.reconstructFromCorrsPy(corrs, calibXMLFiles, keypointsReconFile, reconConfig)
#         # MultiCameraReconstructFromRecog.outputCorrespondencesPy(corrs, calibXMLFiles, keypointsReconFile, reconConfig)
#
#         # triangulate using python code
#         triangulations = []
#         for i in range(len(corrs[0])):
#             camPts = []
#             selectedCamProjMats = []
#             confidence = []
#             for iCam in range(len(camProjMats)):
#                 if corrs[iCam][i][0] != -1:
#
#                     camPts.append(corrs[iCam][i])
#                     confidence.append(corrs[iCam][i][2])
#                     selectedCamProjMats.append(camProjMats[iCam])
#
#             if numMostConfidentToPick != -1:
#                 sortedId = np.argsort(-np.array(confidence))
#                 # print(sortedId)
#                 selecedIds = sortedId[:numMostConfidentToPick]
#                 camPts = [camPts[i] for i in selecedIds]
#                 selectedCamProjMats = [selectedCamProjMats[i] for i in selecedIds]
#
#             try:
#                 keyPoint3D, errs = Triangulation.mulCamsDLT(camPts, selectedCamProjMats)
#
#                 print(errs.shape[0], errs)
#
#                 if np.mean(errs) < reprojectErrThreshold:
#                     triangulations.append(keyPoint3D)
#                 else:
#                     triangulations.append([0,0,-1])
#             except:
#                 triangulations.append([0,0,-1])
#         if drawReconstructedKeypoints:
#             for iCam, (img, imgF) in enumerate( zip(imgs, undistImgFiles)):
#                 outResultOImgFile = join(folderProjectedKP, Path(imgF).stem + '.png')
#                 project3DKeypoints(outResultOImgFile, img, camProjMats[iCam], triangulations, keypointSkeletonParentTable)
#
#
#         # reconstructionData = json.load(open(keypointsReconFile))
#         # triangulations = reconstructionData['Triangulation']
#
#         outTriangulationObjFile = join(reconFolder, "PointCloud.obj")
#         write_obj(outTriangulationObjFile, triangulations)

