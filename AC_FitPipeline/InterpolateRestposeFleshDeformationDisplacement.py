# from InterpolateRestPose import *
import glob
from pathlib import Path
import json
import tqdm
import scipy.sparse
import scipy.sparse.linalg
import time
import subprocess

from pypardiso import spsolve
from SkelFit import Data
from SkelFit import Visualization
from SkelFit.PoseBlendShape import *

import shutil

def makeTemporalLaplacainBlock(dim):
    TL = sparse.lil_matrix((3 * dim, 3 * dim))
    I = sparse.eye(dim)
    TL[:dim, :dim] = I
    TL[dim:2 * dim, :dim] = -2 * I
    TL[2 * dim:3 * dim, :dim] = I

    TL[:dim, dim:2 * dim] = -2 * I
    TL[dim:2 * dim, dim:2 * dim] = 4 * I
    TL[2 * dim:3 * dim, dim:2 * dim] = -2 * I

    TL[:dim, 2 * dim:3 * dim] = I
    TL[dim:2 * dim, 2 * dim:3 * dim] = -2 * I
    TL[2 * dim:3 * dim, 2 * dim:3 * dim] = I

    # Actually that is negative temporal laplacian
    # No it is not
    # because the inner product is even function, it does matter
    # TL = -1*TL

    return TL

def makeRegualrTemporalLaplacainBlock(dim):
    TL = sparse.lil_matrix((2 * dim, 2 * dim))
    I = sparse.eye(dim)
    TL[:dim, :dim] = I
    TL[dim:2 * dim, :dim] = -1 * I

    TL[:dim, dim:2 * dim] = -1 * I
    TL[dim:2 * dim, dim:2 * dim] = I

    return TL

def interpolateRestPoseDeformationWithTemporalCoherence(restPoseTarget, inSkelDataFile, inFaceMeshPath,outFolder, interpolationOverlappingLength = 30,
                                                        interpolationSegLength=100, tw=1, interval=[], inputExt = 'obj', chunkedInput=False, blendOverlapping=True):
    os.makedirs(outFolder, exist_ok=True)

    restPosePts = np.array(json.load(open(inSkelDataFile))["VTemplate"])

    restPosePts = np.transpose(restPosePts[0:3, :])
    meshWithFaces = pv.PolyData(inFaceMeshPath)
    meshWithFaces.points = restPosePts

    meshWithFaces.save('LaplacianMesh.ply', binary=False)

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    UV = igl.eigen.MatrixXd()
    U = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readPLY('LaplacianMesh.ply', V, F, N, UV)

    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V, F, L)

    LSBlock = -e2p(L)

    meshWithFaces = pv.PolyData(inFaceMeshPath)
    meshWithFaces.points = restPosePts

    allCapturePts = []
    # Load Data
    if not chunkedInput:
        objFiles = glob.glob(restPoseTarget + r'\*.' + inputExt)
        if len(interval) == 2:
            objFiles = objFiles[interval[0]:interval[1]]
        for f in tqdm.tqdm(objFiles, desc="Load Data"):
            mesh = pv.PolyData(f)
            allCapturePts.append(mesh.points)

    else:
        scanData = json.load(open(restPoseTarget))
        for data in scanData:
            allCapturePts.append(data["Pts"])
            objFiles = data["BatchFiles"]

    numFrames = len(allCapturePts)
    numCaptureDataDims = allCapturePts[0].shape[0]
    numMeshVertsDims = mesh.points.shape[0]
    allCapturePts = np.vstack(allCapturePts)

    # Build interpolation laplacian matrix
    LSBlock = sparse.csc_matrix(LSBlock)

    L = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(interpolationSegLength), desc="Build Spacial Laplacian"):
        L[i * numMeshVertsDims:(i + 1) * numMeshVertsDims, i * numMeshVertsDims:(i + 1) * numMeshVertsDims] = LSBlock

    TLBlock = makeTemporalLaplacainBlock(numMeshVertsDims)
    TLBlock = sparse.csc_matrix(TLBlock)

    TL = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(1, interpolationSegLength - 1), desc="Build Temporal Laplacian"):
        TL[(i - 1) * numMeshVertsDims:(i + 2) * numMeshVertsDims, (i - 1) * numMeshVertsDims:(i + 2) * numMeshVertsDims] += TLBlock

    # Blend spacial laplacian and temporal laplacian
    if tw:
        LAll = L + tw * TL
    else:
        LAll = L

    interpoStep = interpolationSegLength - interpolationOverlappingLength

    mesh = pv.PolyData(inFaceMeshPath)

    restPosePts = np.array(json.load(open(inSkelDataFile))["VTemplate"])
    restPosePts = np.transpose(restPosePts[0:3, :])

    restPoseStacked = np.vstack([restPosePts for i in range(interpolationSegLength)])

    xInterpoLast = None

    # Do interpolation in step of interpolationSegLength
    for iStep in tqdm.tqdm(range(int(numFrames / interpoStep)-2), desc="Do interpolation in steps"):
        iStart = iStep * interpoStep * numMeshVertsDims
        iEnd = iStart + interpolationSegLength * numMeshVertsDims

        X = allCapturePts[iStart:iEnd, 0] - restPoseStacked[:, 0]
        Y = allCapturePts[iStart:iEnd, 1] - restPoseStacked[:, 1]
        Z = allCapturePts[iStart:iEnd, 2]
        observedIds = np.where(Z != -1)[0]
        nConstraints = observedIds.shape[0]
        Z = Z - restPoseStacked[:, 2]

        nDimX = X.shape[0]

        A = LAll.copy()
        # A = L.copy()
        A.resize(LAll.shape[0] + nConstraints, LAll.shape[1] + nConstraints)

        # L[nDimX:nConstraints + nDimX, 0:nDimX] = D
        # L[0:nDimX, nDimX:nConstraints + nDimX] = D.transpose()

        KKTRes = sparse.lil_matrix((nDimX + nConstraints, 3))
        A = sparse.lil_matrix(A)
        for i, vId in enumerate(observedIds):
            KKTRes[i + nDimX, 0] = X[vId]
            KKTRes[i + nDimX, 1] = Y[vId]
            KKTRes[i + nDimX, 2] = Z[vId]
            A[i + nDimX, vId] = 1
            A[vId, i + nDimX] = 1
        # A = sparse.csc_matrix(A)
        # KKTRes = sparse.csc_matrix(KKTRes)
        A = sparse.csr_matrix(A)
        KKTRes = sparse.csr_matrix(KKTRes)
        # print("Done building system")

        # xInterpo = sparse.linalg.spsolve(A, KKTRes)
        xInterpo = spsolve(A, KKTRes)
        xInterpo = xInterpo.toarray()[:interpolationSegLength * numCaptureDataDims, :] + restPoseStacked


        if blendOverlapping and iStep != 0:
            blendWeights = [i / interpolationOverlappingLength for i in range(1, interpolationOverlappingLength + 1)]
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])

                if iF < interpolationOverlappingLength:
                    mesh.points = blendWeights[iF] * xInterpo[iF * numCaptureDataDims:(iF + 1) * numCaptureDataDims, :] + \
                    (1 - blendWeights[iF]) * xInterpoLast[(iF + interpoStep)* numCaptureDataDims:(iF + interpoStep + 1) * numCaptureDataDims, :]

                else:
                    mesh.points = xInterpo[iF * numCaptureDataDims:(iF + 1) * numCaptureDataDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)


        else:
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])
                mesh.points = xInterpo[iF * numCaptureDataDims:(iF + 1) * numCaptureDataDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)

        xInterpoLast = xInterpo

def interpolateRestPoseDeformationWithTemporalCoherence2DifferentMesh(restPoseTarget, inSkelDataFile, inFaceMeshPath, outFolder, interpolationOverlappingLength = 30,
                                                        interpolationSegLength=100, tw=1, interval=[], inputExt = 'obj', chunkedInput=False, blendOverlapping=True):
    os.makedirs(outFolder, exist_ok=True)

    meshFilename, meshFileExtension = os.path.splitext(inFaceMeshPath)
    triVids = np.array(json.load(open(inSkelDataFile))["TriVidsNp"])

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    UV = igl.eigen.MatrixXd()
    U = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    if meshFileExtension.lower() == '.ply':
        igl.readPLY(inFaceMeshPath, V, F, N, UV)
    elif meshFileExtension.lower() == '.obj':
        igl.readOBJ(inFaceMeshPath, V, F)
    else:
        print("Unsupported file format: ", meshFileExtension)

    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V, F, L)

    LSBlock = -e2p(L)

    allCapturePts = []
    # Load Data
    if not chunkedInput:
        objFiles = glob.glob(restPoseTarget + r'\*.' + inputExt)
        if len(interval) == 2:
            objFiles = objFiles[interval[0]:interval[1]]
        for f in tqdm.tqdm(objFiles, desc="Load Data"):
            mesh = pv.PolyData(f)
            allCapturePts.append(mesh.points)

    else:
        scanData = json.load(open(restPoseTarget))
        for data in scanData:
            allCapturePts.append(np.array(data["Pts"]))
            objFiles = data["BatchFiles"]

    resposeMesh = pv.PolyData(inFaceMeshPath)

    numFrames = len(allCapturePts)
    numCaptureDataDims = allCapturePts[0].shape[0]
    numMeshVertsDims = resposeMesh.points.shape[0]

    # pad every point cloud in allCapturePts to have numMeshVertsDims vertices, treat the padded points as never observed points
    for i in range(len(allCapturePts)):
        paddedPts = np.zeros((numMeshVertsDims, 3))
        paddedPts[:numCaptureDataDims,:] = allCapturePts[i]
        paddedPts[numCaptureDataDims:, 2] = -1
        allCapturePts[i] = paddedPts

    allCapturePts = np.vstack(allCapturePts)

    # Build interpolation laplacian matrix
    LSBlock = sparse.csc_matrix(LSBlock)

    maxVal = np.max(np.abs(LSBlock.toarray()))
    print("Max value in Spatial Laplacian:", maxVal )
    assert maxVal < 10

    L = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(interpolationSegLength), desc="Build Spacial Laplacian"):
        L[i * numMeshVertsDims:(i + 1) * numMeshVertsDims, i * numMeshVertsDims:(i + 1) * numMeshVertsDims] = LSBlock

    TLBlock = makeTemporalLaplacainBlock(numMeshVertsDims)
    TLBlock = sparse.csc_matrix(TLBlock)

    TL = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(1, interpolationSegLength - 1), desc="Build Temporal Laplacian"):
        TL[(i - 1) * numMeshVertsDims:(i + 2) * numMeshVertsDims,
        (i - 1) * numMeshVertsDims:(i + 2) * numMeshVertsDims] += TLBlock

    # Blend spacial laplacian and temporal laplacian
    LAll = L + tw * TL

    interpoStep = interpolationSegLength - interpolationOverlappingLength

    mesh = pv.PolyData(inFaceMeshPath)

    # restPosePts = np.array(json.load(open(inSkelDataFile))["VTemplate"])
    # restPosePts = np.transpose(restPosePts[0:3, :])

    restPosePts = mesh.points

    restPoseStacked = np.vstack([restPosePts for i in range(interpolationSegLength)])

    xInterpoLast = None

    # for deprecated isolated vertex on rest pose we need to set it fixed, otherwise KKT becomes singular)
    depracatedVertsOnRestpose = np.where(restPoseStacked[:, 2] == -1)[0]

    # Do interpolation in step of interpolationSegLength
    numSteps = int((numFrames - interpolationSegLength) / interpoStep) + 1
    for iStep in tqdm.tqdm(range(numSteps), desc="Do interpolation in steps"):
        timeStart = time.clock()
        iStart = iStep * interpoStep * numMeshVertsDims
        iEnd = iStart + interpolationSegLength * numMeshVertsDims

        X = allCapturePts[iStart:iEnd, 0] - restPoseStacked[:, 0]
        Y = allCapturePts[iStart:iEnd, 1] - restPoseStacked[:, 1]
        Z = allCapturePts[iStart:iEnd, 2]
        observedIds = np.where(Z != -1)[0]

        constraintIds = np.union1d(observedIds, depracatedVertsOnRestpose)

        nConstraints = constraintIds.shape[0]
        Z = Z - restPoseStacked[:, 2]

        nDimX = LAll.shape[0]

        A = LAll.copy()
        # A = L.copy()
        A.resize(LAll.shape[0] + nConstraints, LAll.shape[1] + nConstraints)

        # L[nDimX:nConstraints + nDimX, 0:nDimX] = D
        # L[0:nDimX, nDimX:nConstraints + nDimX] = D.transpose()

        KKTRes = sparse.lil_matrix((LAll.shape[0] + nConstraints, 3))
        A = sparse.lil_matrix(A)
        for i, vId in enumerate(constraintIds):
            # for deprecated isolated vertex on rest pose we need to set it fixed, otherwise KKT becomes singular)
            if vId in depracatedVertsOnRestpose:
                KKTRes[i + nDimX, 0] = 0
                KKTRes[i + nDimX, 1] = 0
                KKTRes[i + nDimX, 2] = -1
            else:
                KKTRes[i + nDimX, 0] = X[vId]
                KKTRes[i + nDimX, 1] = Y[vId]
                KKTRes[i + nDimX, 2] = Z[vId]
            A[i + nDimX, vId] = 1
            A[vId, i + nDimX] = 1

        # A = sparse.csc_matrix(A)
        # KKTRes = sparse.csc_matrix(KKTRes)
        A = sparse.csr_matrix(A)
        KKTRes = sparse.csr_matrix(KKTRes)
        # print("Done building system")
        print("Time build KKT:", time.clock() - timeStart)

        # print("Done building system")

        # xInterpo = sparse.linalg.spsolve(A, KKTRes)
        xInterpo = spsolve(A, (KKTRes).toarray())
        xInterpo = xInterpo[:interpolationSegLength * numMeshVertsDims, :]
        print("Time solving linear system:", time.clock()-timeStart)

        print("Spatial Laplacian:", xInterpo.transpose() @ L @ xInterpo)
        print("Temporal Laplacian:", xInterpo.transpose() @ TL @ xInterpo)

        xInterpo = xInterpo + restPoseStacked

        # xInterpo = xInterpo.toarray()[:interpolationSegLength * numMeshVertsDims, :] + restPoseStacked

        if blendOverlapping and iStep != 0 and iStep != numSteps-1:
            blendWeights = [i / interpolationOverlappingLength for i in range(1, interpolationOverlappingLength + 1)]
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])

                if iF < interpolationOverlappingLength:
                    mesh.points = blendWeights[iF] * xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims,:] + \
                                  (1 - blendWeights[iF]) * xInterpoLast[(iF + interpoStep) * numMeshVertsDims:(iF + interpoStep + 1) * numMeshVertsDims,:]

                else:
                    mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)
        elif blendOverlapping and iStep == numSteps-1:
            for iF in range(interpolationSegLength):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])

                if iF < interpolationOverlappingLength:
                    mesh.points = blendWeights[iF] * xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims,:] + \
                                  (1 - blendWeights[iF]) * xInterpoLast[(iF + interpoStep) * numMeshVertsDims:(iF + interpoStep + 1) * numMeshVertsDims,:]

                else:
                    mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)
        else:
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])
                mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)

        xInterpoLast = xInterpo

def interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh(restPoseTarget, inSkelDataFile, inFaceMeshPath, outFolder, interpolationOverlappingLength = 30,
                                                        interpolationSegLength=100, tw=1, interval=[], inputExt = 'obj', chunkedInput=False, blendOverlapping=True, spatialLap=True,
                                                        poseBlendShape=None, quaternions=None, spatialBiLap = False, spatialLapFromSkelData=True):
    os.makedirs(outFolder, exist_ok=True)

    shutil.copy(inSkelDataFile, join(outFolder, 'SkelData.json'))

    skelData = json.load(open(inSkelDataFile))
    restPosePts = (np.array(skelData['VTemplate']).transpose())[:,:3]

    if poseBlendShape is not None:
        restposeWithBSFolder = join(outFolder, 'RestposeWithPoseBS')
        os.makedirs(restposeWithBSFolder, exist_ok=True)

    meshFilename, meshFileExtension = os.path.splitext(inFaceMeshPath)
    triVids = np.array(json.load(open(inSkelDataFile))["TriVidsNp"])

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    UV = igl.eigen.MatrixXd()
    U = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    if spatialLapFromSkelData:

        V = p2e(restPosePts)

        F = p2e(np.array(skelData['Faces']))
    else:
        if meshFileExtension.lower() == '.ply':
            igl.readPLY(inFaceMeshPath, V, F, N, UV)
        elif meshFileExtension.lower() == '.obj':
            igl.readOBJ(inFaceMeshPath, V, F)
        else:
            print("Unsupported file format: ", meshFileExtension)

    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V, F, L)

    LSBlock = -e2p(L)

    allCapturePts = []
    # Load Data
    if not chunkedInput:
        objFiles = glob.glob(restPoseTarget + r'\*.' + inputExt)
        if len(interval) == 2:
            objFiles = objFiles[interval[0]:interval[1]]
        for f in tqdm.tqdm(objFiles, desc="Load Data"):
            mesh = pv.PolyData(f)
            allCapturePts.append(mesh.points)

    else:
        scanData = json.load(open(restPoseTarget))
        for data in scanData:
            allCapturePts.append(np.array(data["Pts"]))
            objFiles = data["BatchFiles"]

    # resposeMesh = pv.PolyData(inFaceMeshPath)

    numFrames = len(allCapturePts)
    numCaptureDataDims = allCapturePts[0].shape[0]
    numMeshVertsDims = restPosePts.shape[0]

    # pad every point cloud in allCapturePts to have numMeshVertsDims vertices, treat the padded points as never observed points
    for i in range(len(allCapturePts)):
        paddedPts = np.zeros((numMeshVertsDims, 3))
        paddedPts[:numCaptureDataDims,:] = allCapturePts[i]
        paddedPts[numCaptureDataDims:, 2] = -1
        allCapturePts[i] = paddedPts

    allCapturePts = np.vstack(allCapturePts)

    # Build interpolation laplacian matrix
    LSBlock = sparse.csc_matrix(LSBlock)


    maxVal = np.max(np.abs(LSBlock.toarray()))
    print("Max value in Spatial Laplacian:", maxVal )
    assert maxVal < 200

    if spatialBiLap:
        print("Using BiLaplcian")
        LSBlock = LSBlock @ LSBlock

    L = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(interpolationSegLength), desc="Build Spacial Laplacian"):
        L[i * numMeshVertsDims:(i + 1) * numMeshVertsDims, i * numMeshVertsDims:(i + 1) * numMeshVertsDims] = LSBlock

    TLBlock = makeRegualrTemporalLaplacainBlock(numMeshVertsDims)
    TLBlock = sparse.csc_matrix(TLBlock)

    TL = sparse.csc_matrix((interpolationSegLength * numMeshVertsDims, interpolationSegLength * numMeshVertsDims))
    for i in tqdm.tqdm(range(1, interpolationSegLength), desc="Build Temporal Laplacian"):
        TL[(i - 1) * numMeshVertsDims:(i + 1) * numMeshVertsDims,
        (i - 1) * numMeshVertsDims:(i + 1) * numMeshVertsDims] += TLBlock

    # Blend spacial laplacian and temporal laplacian

    if spatialLap:
        LAll = L + tw * TL

        # if tw != 0:
        #     LAll = L + tw * TL
        # else:
        #     LAll = L
    else:
        LAll = tw * TL

    interpoStep = interpolationSegLength - interpolationOverlappingLength

    mesh = pv.PolyData(inFaceMeshPath)

    # Rest pose verts better come from Skel data

    restPoseStacked = np.vstack([restPosePts for i in range(interpolationSegLength)])
    xInterpoLast = None

    # for deprecated isolated vertex on rest pose we need to set it fixed, otherwise KKT becomes singular)
    depracatedVertsOnRestpose = np.where(restPoseStacked[:, 2] == -1)[0]

    # Do interpolation in step of interpolationSegLength
    numSteps = int((numFrames - interpolationSegLength) / interpoStep) + 1
    for iStep in tqdm.tqdm(range(numSteps), desc="Do interpolation in steps"):

        timeStart = time.clock()
        iStart = iStep * interpoStep * numMeshVertsDims
        iEnd = iStart + interpolationSegLength * numMeshVertsDims

        # if pose blend shape is not none then apply pose blend Shape
        if poseBlendShape is not None:
            # quaternions, trans = Data.readBatchedSkelParams(poseParamFile)
            RAllFrames = [quaternionsToRotations(q16) for q16 in quaternions]
            restposeWithBlendShapes = [applyPoseBlendShapes(restPosePts, poseBlendShape, R) for R in RAllFrames[iStep * interpoStep:iStep * interpoStep + interpolationSegLength]]
            restPoseStacked = np.vstack(restposeWithBlendShapes)

            # write the rest pose with pose blend shape
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])

                restposeWithPBS = restposeWithBlendShapes[iF]

                mesh.points = restposeWithPBS
                mesh.save(restposeWithBSFolder + '\\' + objFp.stem + '.ply', binary=False)

        X = allCapturePts[iStart:iEnd, 0] - restPoseStacked[:, 0]
        Y = allCapturePts[iStart:iEnd, 1] - restPoseStacked[:, 1]
        Z = allCapturePts[iStart:iEnd, 2]
        observedIds = np.where(Z != -1)[0]

        constraintIds = np.union1d(observedIds, depracatedVertsOnRestpose)

        nConstraints = constraintIds.shape[0]
        Z = Z - restPoseStacked[:, 2]

        nDimX = LAll.shape[0]

        A = LAll.copy()
        # A = L.copy()
        A.resize(LAll.shape[0] + nConstraints, LAll.shape[1] + nConstraints)

        KKTRes = sparse.lil_matrix((LAll.shape[0] + nConstraints, 3))
        A = sparse.lil_matrix(A)
        for i, vId in enumerate(constraintIds):
            # for deprecated isolated vertex on rest pose we need to set it fixed, otherwise KKT becomes singular)
            if vId in depracatedVertsOnRestpose:
                KKTRes[i + nDimX, 0] = 0
                KKTRes[i + nDimX, 1] = 0
                KKTRes[i + nDimX, 2] = 0
            else:
                KKTRes[i + nDimX, 0] = X[vId]
                KKTRes[i + nDimX, 1] = Y[vId]
                KKTRes[i + nDimX, 2] = Z[vId]
            A[i + nDimX, vId] = 1
            A[vId, i + nDimX] = 1

        A = sparse.csr_matrix(A)
        KKTRes = sparse.csr_matrix(KKTRes)
        # print("Done building system")
        print("Time build KKT:", time.clock() - timeStart)

        # xInterpo = sparse.linalg.spsolve(A, KKTRes)
        xInterpo = spsolve(A, (KKTRes).toarray())
        xInterpo = xInterpo[:interpolationSegLength * numMeshVertsDims, :]
        print("Time solving linear system:", time.clock()-timeStart)

        # print("Spatial Laplacian:", xInterpo.transpose() @ L @ xInterpo)
        # print("Temporal Laplacian:", xInterpo.transpose() @ TL @ xInterpo)
        xInterpo = xInterpo + restPoseStacked

        blendWeights = [i / interpolationOverlappingLength for i in range(1, interpolationOverlappingLength + 1)]
        if blendOverlapping and iStep != 0 and iStep != numSteps-1:
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])

                if iF < interpolationOverlappingLength:
                    mesh.points = blendWeights[iF] * xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims,:] + \
                                  (1 - blendWeights[iF]) * xInterpoLast[(iF + interpoStep) * numMeshVertsDims:(iF + interpoStep + 1) * numMeshVertsDims,:]

                else:
                    mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)
        elif blendOverlapping and iStep == numSteps-1:
            if iStep != 0:
                for iF in range(interpolationSegLength):
                    fid = iStep * interpoStep + iF
                    objFp = Path(objFiles[fid])

                    if iF < interpolationOverlappingLength:
                        mesh.points = blendWeights[iF] * xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims,:] + \
                                      (1 - blendWeights[iF]) * xInterpoLast[(iF + interpoStep) * numMeshVertsDims:(iF + interpoStep + 1) * numMeshVertsDims,:]

                    else:
                        mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                    mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)
            else:
                # only 1 step in this case
                for iF in range(interpolationSegLength):
                    fid = iStep * interpoStep + iF
                    objFp = Path(objFiles[fid])
                    mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                    mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)
        else:
            for iF in range(interpoStep):
                fid = iStep * interpoStep + iF
                objFp = Path(objFiles[fid])
                mesh.points = xInterpo[iF * numMeshVertsDims:(iF + 1) * numMeshVertsDims, :]
                mesh.save(outFolder + '\\' + objFp.stem + '.ply', binary=False)



        xInterpoLast = xInterpo

def interpolateRestPoseDeformation(restPoseTargetFolder, inSkelDataFile, inFaceMeshPath, outFolder):
    os.makedirs(outFolder, exist_ok=True)
    objFiles = glob.glob(restPoseTargetFolder + r'\*.obj')

    restPosePts = np.array(json.load(open(inSkelDataFile))["VTemplate"])
    restPosePts = np.transpose(restPosePts[0:3, :])
    meshWithFaces = pv.PolyData(inFaceMeshPath)
    meshWithFaces.points = restPosePts

    meshWithFaces.save('LaplacianMesh.ply', binary=False)

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    UV = igl.eigen.MatrixXd()
    U = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readPLY('LaplacianMesh.ply', V, F, N, UV)

    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V, F, L)

    LNP = -e2p(L).todense()

    meshWithFaces = pv.PolyData(inFaceMeshPath)
    meshWithFaces.points = restPosePts

    for f in tqdm.tqdm(objFiles):
        mesh = pv.PolyData(f)

        obsVerts = np.where(mesh.points[:, 2] != -1)[0]

        nDimX = mesh.points.shape[0]
        nConstraints = obsVerts.shape[0]

        for iDim in range(3):
            x = mesh.points[:, iDim] - meshWithFaces.points[:, iDim]
            # Build Constraint
            D = np.zeros((nConstraints, nDimX))
            e = np.zeros((nConstraints, 1))
            for i, vId in enumerate(obsVerts):
                D[i, vId] = 1
                e[i, 0] = x[vId]

            kMat, KRes = buildKKT(LNP, D, e)
            xInterpo = np.linalg.solve(kMat, KRes)
            mesh.points[:, iDim] = xInterpo[0:nDimX, 0] + meshWithFaces.points[:, iDim]

        fp = Path(f)
        mesh.faces = meshWithFaces.faces
        mesh.save(outFolder + '\\' + fp.stem + r'.ply', binary=False)

def prepareRestPoseDeformatoinBatch(dataFolder, outFolder, ext = 'ply'):
    os.makedirs(outFolder, exist_ok=True)

    files = glob.glob(dataFolder + r'\*.' + ext)

    deformedRestPoseFiles =[]
    for f in files:
        mesh = pv.PolyData(f)
        points = mesh.points
        nPts = points.shape[0]
        points = np.vstack([np.transpose(points), np.ones((1, nPts))])

        fp = Path(f)
        outFile = outFolder + '\\' + fp.stem + '.json'

        json.dump({'Pts':points.tolist()}, open(outFile, 'w'), indent=2)
        deformedRestPoseFiles.append(outFile)

    outBatchFile = dataFolder + r'\BatchFile.json'
    json.dump({'BatchFiles':deformedRestPoseFiles}, open(outBatchFile, 'w'), indent=2)
    return outBatchFile

if __name__ == '__main__':

    TL = makeRegualrTemporalLaplacainBlock(3)
    TL =  makeTemporalLaplacainBlock(3)
    TL = TL.toarray()
    print(TL)

    # dataFolder = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal5ToNewTriangulation"
    # dataFolder = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal6ToNewTriangulationOutlierFiltered\RestPoseTarget"
    # restPoseTarget = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal6ToNewTriangulationOutlierFiltered\RestPoseTarget'
    # outFolder = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptSequence\Expriment2\Fit1e-4Normalized\RestPoseDetails\InterpolationDisplacement'

    # dataFolder = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal7ToNewTriangulationOutlierFiltered40\RestPoseTarget"
    # restPoseTarget = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal7ToNewTriangulationOutlierFiltered40\RestPoseTarget'
    #
    # # inFaceMeshPath = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'
    # inFaceMeshPath = r'D:\GDrive\mocap\2019_09_10_KutayFixedMesh1487\simpmeshwithfeetneworderv3.ply'
    # inSkelDataFile = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\SKelModel\Skel0040Interpolated.json'
    #
    # # deformedBatchFile = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal6ToNewTriangulationOutlierFiltered\RestPoseTarget\InterpolationDisplacement\BatchFile.json"
    # paramJsonFile = r"F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal7ToNewTriangulationOutlierFiltered40\WithTC\Params\0Poses.json"
    #
    # chunkedInput = False
    # skipInterpo = True
    # # skipInterpo = False
    # inputExt = 'obj'
    # # interval = [0, 1000]
    # # interval = [0, 100]
    # interval = []
    # interpolationSegLength = 100
    # interpolationOverlappingLength = 30
    #
    # temporalWeights = [1000]
    #
    # for tw in temporalWeights:
    #     outFolder = dataFolder + r'\Test\TW' + str(tw)
    #
    #     if not skipInterpo:
    #         interpolateRestPoseDeformationWithTemporalCoherence(restPoseTarget, inSkelDataFile, inFaceMeshPath, outFolder, tw=tw, interval=interval,
    #             interpolationSegLength=interpolationSegLength, interpolationOverlappingLength = interpolationOverlappingLength, blendOverlapping=True)
    #
    #     motionFolder= os.path.join(outFolder, "Motion")
    #     os.makedirs(motionFolder, exist_ok=True)
    #     deformedBatchFile = prepareRestPoseDeformatoinBatch(outFolder, outFolder + r'\Batch')
    #     recoverCmd = ['RecoverFleshDeformations', deformedBatchFile, paramJsonFile, motionFolder, '-s',
    #          inSkelDataFile]
    #     print(*recoverCmd)
    #     subprocess.call(recoverCmd)

    # # dataFolder = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptSequence\Expriment2\Fit1e-4Normalized\RestPoseDetails'
    # dataFolder = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\FitFinal\TargetPtsBackToRestPose'
    # # outFolder = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptSequence\Expriment2\Fit1e-4Normalized\RestPoseDetails\InterpolationDisplacement'
    # outFolder = dataFolder + r'\InterpolationDisplacement'
    #
    # # inFaceMeshPath = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\AlternateOptimization\cornersRegisteredToSMpl.ply'
    # inOriginalRestPoseMesh = r'D:\GDrive\mocap\2019_09_10_KutayFixedMesh1487\simpmeshwithfeetneworderv3.ply'
    # inSkelDataFile = r'F:\WorkingCopy2\2019_08_09_AllPoseCapture\OptRestPose\Expriment8\SKelModel\Skel0040Interpolated.json'
    #
    # inFaceMeshPath = inOriginalRestPoseMesh
    #
    # os.makedirs(outFolder, exist_ok=True)
    #
    # objFiles = glob.glob(dataFolder + r'\*.obj')
    #
    # V = igl.eigen.MatrixXd()
    # N = igl.eigen.MatrixXd()
    # UV = igl.eigen.MatrixXd()
    # U = igl.eigen.MatrixXd()
    # F = igl.eigen.MatrixXi()
    # igl.readPLY(inOriginalRestPoseMesh, V, F, N, UV)
    #
    # L = igl.eigen.SparseMatrixd()
    # igl.cotmatrix(V, F, L)
    #
    # LNP = e2p(L).todense()
    #
    # restPosePts = np.array(json.load(open(inSkelDataFile))["VTemplate"])
    # restPosePts = np.transpose(restPosePts[0:3, :])
    #
    # meshWithFaces = pv.PolyData(inFaceMeshPath)
    # meshOriginal = pv.PolyData()
    # meshOriginal.points = restPosePts
    #
    # for f in objFiles:
    #     mesh = pv.PolyData(f)
    #
    #     obsVerts = np.where(mesh.points[:, 2] != -1)[0]
    #
    #     nDimX = mesh.points.shape[0]
    #     nConstraints = obsVerts.shape[0]
    #
    #     for iDim in range(3):
    #         x = mesh.points[:, iDim] - meshOriginal.points[:, iDim]
    #         # Build Constraint
    #         D = np.zeros((nConstraints, nDimX))
    #         e = np.zeros((nConstraints, 1))
    #         for i, vId in enumerate(obsVerts):
    #             D[i, vId] = 1
    #             e[i, 0] = x[vId]
    #
    #         kMat, KRes = buildKKT(LNP, D, e)
    #         xInterpo = np.linalg.solve(kMat, KRes)
    #         mesh.points[:, iDim] = xInterpo[0:nDimX, 0] + meshOriginal.points[:, iDim]
    #
    #     fp = Path(f)
    #     mesh.faces = meshWithFaces.faces
    #     mesh.save(outFolder + '\\' + fp.stem + r'.vtk')