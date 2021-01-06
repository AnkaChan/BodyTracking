from SkelFit.Debug import *
from SkelFit.Visualization import *
from SkelFit.Data import *
import os
import json
from pathlib import Path
import pyigl as igl
from iglhelpers import *
import shutil
from pypardiso import spsolve
import time

class Config:
    def __init__(self):
        # overall control
        self.fitInterval = []
        self.nameOutFolerUsingParams = True
        self.doInitialFit = True
        self.optimizeJointAngleTemporalCoherence = True
        self.mapToRestPose = True
        self.detailRecover = True

        self.removeOutliers = False
        self.addGround = False

        # for initial & temporal smoothing fit
        ## for initial fit
        self.robustifierThreshold = 30
        self.followingFunctionTolerance = 1e-3
        self.poseChangeRegularizer = True
        self.poseChangeRegularizerWeight = 400
        ## for temporal smoothing fit
        self.jointTCW = 0.5
        self.jointBiLap = 0
        self.numPtsCompleteMesh = 1626
        self.posePreservingTerm = True # to preserve the input pose
        self.posePreservingWeight = 500

        # Detail Interpolation
        self.interpolationSegLength=None
        self.fitInterval = []
        self.tw = 1
        self.interpolationSegLength = 50
        self.interpolationOverlappingLength = 0
        self.spatialLap = True
        self.spatialBiLap = True
        self.spatialLapFromSkelData = False

        # visualization
        self.visualizeTargets = True
        self.visualizeRestposeTargets = True
        self.visualizeCorrs = True
        self.visualizeInitialFit = False
        self.visualizeTCFit = True
        self.visualizeDisplacementInterpo = True


        self.outlierIds = []

        self.usePoseBlendShape = False
        self.externalTLWeight = None

def removeOutliers(inChunkFile, newChunkFile, outliers):
    data = json.load(open(inChunkFile))
    pts = np.array(data['Pts'])

    for outlierGroup in outliers:
        for vId in outlierGroup["Ids"]:
            pts[outlierGroup["Frames"], vId, :] = [0,0,-1]

    data['Pts'] = pts.tolist()

    json.dump(data, open(newChunkFile, 'w'), indent=2)
    return newChunkFile

def getFitName(cfg):
    fitName = ('SLap_' + 'SBiLap_' + str(cfg.spatialBiLap) + '_') if cfg.spatialLap else ''

    fitName = fitName + 'TLap_' + str(cfg.tw)

    fitName = fitName + "_JTW_" + str(cfg.jointTCW) + '_JBiLap_' + str(cfg.jointBiLap) \
              + '_Step' + str(cfg.interpolationSegLength) + '_Overlap' + str(cfg.interpolationOverlappingLength)
    if cfg.removeOutliers:
        fitName = fitName + '_' + 'cleaned'

    if cfg.usePoseBlendShape:
        fitName = fitName + '_PBS'

    return fitName

def lbsFitting(inChunkFile, outputDataFolder, inSkelDataFile, cfg = Config()):

    if cfg.removeOutliers:
        outlierRemoveChunkFile = inChunkFile + '.cleaned.json'
        inChunkFile = removeOutliers(inChunkFile, outlierRemoveChunkFile, cfg.outlierIds)

    if cfg.nameOutFolerUsingParams:
        fitName = getFitName(cfg)
        outputDataFolder = join(outputDataFolder, fitName)

    fittingDataFolder = outputDataFolder + r'\Init'
    fittingTCFolder = outputDataFolder + r'\LBSWithTC'

    os.makedirs(outputDataFolder, exist_ok=True)
    os.makedirs(fittingDataFolder, exist_ok=True)
    os.makedirs(fittingTCFolder, exist_ok=True)

    # save the parameters for fitting
    outCfgFile = join(outputDataFolder, 'Config.json')
    json.dump(cfg.__dict__, open(outCfgFile, 'w'), indent=2)

    # Do the fit on the motion sequence
    if cfg.doInitialFit:
        print("Do the fit on the motion sequence")

        if len(cfg.fitInterval) == 2:
            fitCmd = ["FitToPointCloudDynamic", inChunkFile, fittingDataFolder, '-b',
                      str(cfg.fitInterval[0]), '-e', str(cfg.fitInterval[1]), '--followingFunctionTolerance',
                      str(cfg.followingFunctionTolerance),
                      '-s', inSkelDataFile, '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '100', '-c',
                      '--outputChunkFile']
        else:
            fitCmd = ["FitToPointCloudDynamic", inChunkFile, fittingDataFolder, '-s', inSkelDataFile,
                      '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '100', '-c', '--outputChunkFile',
                      '--followingFunctionTolerance', str(cfg.followingFunctionTolerance)]

        if cfg.poseChangeRegularizer:
            fitCmd.extend(
                ['--poseChangeRegularizer', '1', '--poseChangeRegularizerWeight', str(cfg.poseChangeRegularizerWeight)])

        print(*fitCmd)
        subprocess.call(fitCmd)

def mapBackToRestpose(targetPCs, inSkelDataFile, fileNames, outFolder, qs, ts):
    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(inSkelDataFile)

    for iF in tqdm.tqdm(range(len(targetPCs)), desc='Mapping back observed points'):
        targetPC = targetPCs[iF]
        q = qs[iF]
        t = np.array(ts[iF][0])
        Rs = quaternionsToRotations(q)
        targetPC = np.array(targetPC)
        vertsBack = backDeformVerts(targetPC, Rs, t, J, weights, kintreeTable, parent)
        vertsBack[np.where(targetPC[:, 2] == -1)[0], :] = [0,0,-1]
        pv.PolyData(vertsBack, ).save(join(outFolder, fileNames[iF] + '.ply'))

def deformInterpolatedVerts(interpolationFolder, deformedFolder, fileNames, inSkelDataFile,  qs, ts):
    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(inSkelDataFile)

    inputFiles = glob.glob(join(interpolationFolder, '*.ply' ))
    for iF in tqdm.tqdm(range(len(inputFiles)), desc='Deforming interpolated rest pose'):
        interpolatedMesh = pv.PolyData(inputFiles[iF])
        q = qs[iF]
        t = np.array(ts[iF][0])
        Rs = quaternionsToRotations(q)
        interpolatedMesh.points = deformVerts(interpolatedMesh.points, Rs, t, J, weights, kintreeTable, parent)
        interpolatedMesh.save(join(deformedFolder, fileNames[iF] + '.ply'))


def detailInterpolation(inChunkFile, outputDataFolder, inSkelDataFile, inPoseBatchFile, cfg = Config()):
    restPoseTargetFolder = outputDataFolder + r'\RestPoseTarget'
    deformedMeshFolder = outputDataFolder + r'\Interpolated'

    os.makedirs(restPoseTargetFolder, exist_ok=True)
    os.makedirs(deformedMeshFolder, exist_ok=True)

    targetPCs = json.load(open(inChunkFile))
    fileNames = [Path(file).stem for file in targetPCs["BatchFiles"]]
    qs, ts = loadPoseChunkFile(inPoseBatchFile)

    # map the observations back
    if cfg.mapToRestPose:
        mapBackToRestpose(targetPCs['Pts'], inSkelDataFile, fileNames, restPoseTargetFolder, qs,ts)

    interpolationFolder = join(restPoseTargetFolder, 'InterpolationDisplacement')

    if cfg.usePoseBlendShape:
        skelData = json.load(open(inSkelDataFile))
        poseBlendShape = np.array(skelData['PoseBlendShapes'])
    else:
        poseBlendShape = None
    if cfg.detailRecover:
        interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh(restPoseTargetFolder,
                                                                                inSkelDataFile,
                                                                                cfg.inOriginalRestPoseMesh,
                                                                                interpolationFolder,
                                                                                tw=cfg.tw,
                                                                                interpolationSegLength=cfg.interpolationSegLength,
                                                                                interpolationOverlappingLength=cfg.interpolationOverlappingLength,
                                                                                chunkedInput=False,
                                                                                spatialLap=cfg.spatialLap,
                                                                                poseBlendShape=poseBlendShape,
                                                                                quaternions=qs,
                                                                                spatialBiLap=cfg.spatialBiLap,
                                                                                spatialLapFromSkelData=cfg.spatialLapFromSkelData,
                                                                                inputExt='ply'
                                                                                )

    # map interpolated points back to deformed configuration
    deformInterpolatedVerts(interpolationFolder, deformedMeshFolder, fileNames, inSkelDataFile, qs, ts)

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