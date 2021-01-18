import cv2
import json
from Utility import *
from SuitCapture import Camera
from SuitCapture import Triangulation
import pyvista as pv

def calibrationParamsToIEMats(camParam, intrinsicSize4by4 = False,):
    rVec = camParam['rvec']

    rMat, _ = cv2.Rodrigues(np.array(rVec))
    T = np.eye(4)

    T[0:3, 0:3] = rMat
    T[0:3, 3:] = np.array(camParam['tvec'])[:, np.newaxis]

    # print(T)
    fx = camParam['fx']
    fy = camParam['fy']
    cx = camParam['cx']
    cy = camParam['cy']
    if intrinsicSize4by4:
        I = np.array([
            [fx, 0.0, cx, 0],
            [0.0, fy, cy, 0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1],
        ])
    else:
        I = np.array([
            [fx, 0.0, cx, ],
            [0.0, fy, cy],
            [0.0, 0.0, 1],
        ])


    return  I, T


def getProjMats(camParams):

    camProjMats = []
    for iCam in range(len(camParams)):
        camParam = camParams[iCam]
        I, E = Camera.calibrationParamsToIEMats(camParam, True)

        projMat = I @ E
        # pts2D = Triangulation.projectPoints(mesh.points, projMat)
        # pts2Ds.append(pts2D)
        camProjMats.append(projMat)

    return camProjMats

def project3DKeypoints(projMat, keypoints3D, ):
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
            keyPts2D.append(ptsP[:2, 0] )
    keyPts2D = np.array(keyPts2D)

    return keyPts2D


if __name__ == '__main__':
    inCorrsFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\CorrsType1Only'
    inTriangulationFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\TriangulationType1Only'
    calibrationDataFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CameraParameters\cam_params.json'
    flowFolder = r'X:\MocapProj\2021_01_16_OpticalFlows2\Lada_Ground\Flow'
    resizeLvl = 0.5
    frames = ['06141']

    camIds = [0, 4, 8, 12]
    camNames = 'ABCDEFGHIJKLMNOP'
    camParams, camNames = Camera.loadCamParams(calibrationDataFile)

    camProjMats = getProjMats(camParams)

    for frame in frames:
        corrFile = join(inCorrsFolder, 'A' + frame.zfill(8) + '.json')
        corrs = json.load(open(corrFile))

        camIdsObserved = corrs['camIdsUsed']

        triangulationFile = join(inTriangulationFolder, 'A' + frame.zfill(8) + '.obj')
        triangulation = pv.PolyData(triangulationFile)

        flow = np.load(join(flowFolder, 'A' + frame + '.npy'))

        for camId in camIds:
            cName = camNames[camId]
            projMat = camProjMats[camId]
            pts2D = project3DKeypoints(projMat, triangulation.points, )

            for iV in range(pts2D.shape[0]):
                if camId in camIdsObserved[iV]:
                    print(flow[int(pts2D[iV, 1]*resizeLvl), int(pts2D[iV,0]*resizeLvl), :])






