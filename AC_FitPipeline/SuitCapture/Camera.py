import cv2
import numpy as np
import glob, sys, os, json
from pathlib import Path


def writeCalibrationXML(xmlFile, intrinsics, dist, extrinsics):
    fs = cv2.FileStorage(xmlFile, cv2.FILE_STORAGE_WRITE)
    fs.write("intrinsics", intrinsics)
    fs.write("dist", np.array([dist[0], dist[1], dist[2], dist[3], dist[4]]))
    fs.write("extrinsics", extrinsics)
    fs.release()

def loadCalibrationXML(xmlFile):
    cv_file = cv2.FileStorage(xmlFile, cv2.FILE_STORAGE_READ)
    I = cv_file.getNode("intrinsics").mat()
    dist = cv_file.getNode("dist").mat()
    E = cv_file.getNode("extrinsics").mat()
    cv_file.release()
    return  I, dist, E

def loadCamParams(cameraFile):
    calib = json.load(open(cameraFile))
    camParams = []
    camNames = []
    for camName, camParam in calib['cam_params'].items():
        camParams.append(camParam)
        camNames.append(camName)

    return camParams, camNames

def cvCalibrationToIntrinsicMat(camParamCV, size4by4 = False, xRescale = 1, yRescale = 1):
    fx = (camParamCV['fx'] * xRescale)
    fy = (camParamCV['fy'] * yRescale)
    cx = (camParamCV['cx'] * xRescale)
    cy = (camParamCV['cy'] * yRescale)

    if size4by4:
        intrinsic_mtx = np.array([
            [fx, 0.0, cx, 0],
            [0.0, fy, cy, 0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1],
        ])
    else:
        intrinsic_mtx = np.array([
            [fx, 0.0, cx, ],
            [0.0, fy, cy],
            [0.0, 0.0, 1],
        ])

    return intrinsic_mtx

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

