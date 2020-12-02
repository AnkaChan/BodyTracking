# This script deals with case that LBS model and the interpolation mesh has different number of vertices

from SkelFit.Debug import *
from SkelFit.Visualization import *
from SkelFit.Data import *
import os
import json
from pathlib import Path
from InterpolateRestposeFleshDeformationDisplacement import interpolateRestPoseDeformation, \
    prepareRestPoseDeformatoinBatch, interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh
from PoseBlendShape import *
from PoseBlendShape import deformWithPoseBlendShapes

class Config:
    def __init__(self):
        self.fitInterval = []
        self.spatialLap = True
        self.robustifierThreshold = 30
        self.tw = 1
        self.interpolationSegLength = 50
        self.interpolationOverlappingLength = 0

        self.spatialBiLap = True
        self.spatialLapFromSkelData = True

        self.followingFunctionTolerance = 1e-3
        self.jointTCW = 0.5
        self.jointBiLap = 0
        self.numPtsCompleteMesh = 1626
        self.poseChangeRegularizer = True
        self.poseChangeRegularizerWeight = 400

        self.fitInterval = []

        self.doInitialFit = True
        self.optimizeJointAngleTemporalCoherence = True
        self.mapToRestPose = True
        self.detailRecover = True

        self.visualizeTargets = True
        self.visualizeRestposeTargets = True
        self.visualizeCorrs = True
        self.visualizeInitialFit = False
        self.visualizeTCFit = True
        self.visualizeDisplacementInterpo = True
        self.removeOutliers = False
        self.addGround = False

        self.outlierIds = []

        self.usePoseBlendShape = False
        self.externalTLWeight = None
        # self.poseBlendShapeFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\FullBodyModelFitToSMPL\PoseBlendShapes.npy'

def removeOutliers(inChunkFile, newChunkFile, outlierIds, frames = None):
    data = json.load(open(inChunkFile))
    if frames is None:
        pts = np.array(data['Pts'])
        pts[:, outlierIds, :] = [0,0,-1]

        data['Pts'] = pts.tolist()
    else:
        pass

    json.dump(data, open(newChunkFile, 'w'), indent=2)
    return newChunkFile


def removeOutliers2(inChunkFile, newChunkFile, outliers):
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

def interpolateDetials(inChunkFile, outputDataFolder, inSkelDataFile, inSkelDataFileComplete, inOriginalRestPoseMesh, inOriginalRestPoseQuadMesh, cfg = Config()):
    fitName = getFitName(cfg)

    if cfg.removeOutliers:
        outlierRemoveChunkFile = inChunkFile + '.cleaned.json'
        inChunkFile = removeOutliers(inChunkFile, outlierRemoveChunkFile, cfg.outlierIds)

    outputDataFolder = join(outputDataFolder, fitName)
    fittingDataFolder = outputDataFolder + r'\Init'
    fittingTCFolder = outputDataFolder + r'\LBSWithTC'
    fittingPoseBSFolder = outputDataFolder + r'\LBSWithPoseBS'
    restPoseTargetFolder = outputDataFolder + r'\RestPoseTarget'
    deformedMeshFolder = outputDataFolder + r'\Deformed'
    visFolder = join(outputDataFolder, 'Vis')

    os.makedirs(outputDataFolder, exist_ok=True)
    os.makedirs(fittingDataFolder, exist_ok=True)
    os.makedirs(fittingTCFolder, exist_ok=True)
    os.makedirs(restPoseTargetFolder, exist_ok=True)
    os.makedirs(deformedMeshFolder, exist_ok=True)
    os.makedirs(fittingPoseBSFolder, exist_ok=True)

    outCfgFile = join(outputDataFolder, 'Config.json')
    json.dump(cfg.__dict__, open(outCfgFile, 'w'), indent=2)

    targetVisFolder = join(visFolder, 'Targets')
    if cfg.visualizeTargets:
        unpackChunkData(inChunkFile, targetVisFolder, outputType='ply')

    # captureData = json.load(open(inChunkFile))
    # pts = np.array(captureData['Pts'])
    # histogram = SampleDataSatistics(pts)

    # Do the fit on the motion sequence
    if cfg.doInitialFit:
        print("Do the fit on the motion sequence")

        if len(cfg.fitInterval) == 2:
            fitCmd = ["CornersFitReducedQuaternionFToF1487V2", inChunkFile, fittingDataFolder, '-b',
                      str(cfg.fitInterval[0]), '-e', str(cfg.fitInterval[1]), '--followingFunctionTolerance', str(cfg.followingFunctionTolerance),
                      '-s', inSkelDataFile, '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '100', '-c']
        else:
            fitCmd = ["CornersFitReducedQuaternionFToF1487V2", inChunkFile, fittingDataFolder, '-s', inSkelDataFile,
                      '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '100', '-c', '--followingFunctionTolerance', str(cfg.followingFunctionTolerance)]



        if cfg.poseChangeRegularizer:
            fitCmd.extend(['--poseChangeRegularizer', '1', '--poseChangeRegularizerWeight', str(cfg.poseChangeRegularizerWeight)])


        print(*fitCmd)
        subprocess.call(fitCmd)

    # make the batch file for fitting parameters and targets
    print("Make the batch file for fitting parameters")
    fittingParamFolder = fittingDataFolder + r"\Params"

    initialParamJsonFile = fittingParamFolder + "\\0Poses.json"
    makeBatchFileFromFolder(fittingParamFolder, 'txt', initialParamJsonFile)

    if cfg.visualizeInitialFit:
        print("Visualize Initial Fit")

        deformedCompleteMeshFolderInit = join(fittingDataFolder, 'CompleteModel')
        os.makedirs(deformedCompleteMeshFolderInit, exist_ok=True)
        if cfg.numPtsCompleteMesh == 1626:
            subprocess.call(
                ['SkeletonDeformation1626', initialParamJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolderInit])
        else:
            print('SkeletonDeformation1692', initialParamJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolderInit)
            subprocess.call(
                ['SkeletonDeformation1692', initialParamJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolderInit])

        initialFitVTKFolder = join(visFolder, 'InitialFit')
        fittingToVtk(deformedCompleteMeshFolderInit, outVTKFolder=initialFitVTKFolder, addABeforeName=True, addGround=cfg.addGround, meshWithFaces=inOriginalRestPoseQuadMesh)
        # visualizeCorrs(targetFiles, fittingDataFolder, fittingDataFolder + r'\Corrs', sanityCheck=False)

    # Run temporal coherence optimization

    paramJsonFolder = fittingTCFolder + "\\Params"
    paramJsonFile = paramJsonFolder + "\\0Poses.json"
    if cfg.optimizeJointAngleTemporalCoherence:
        print("Run temporal coherence optimization")
        tpcCmd = ['CornersFitReducedBatchedQuaternionTCWithInit2', inChunkFile, initialParamJsonFile, fittingTCFolder,
                  '-s',
                  inSkelDataFile, '-j', str(cfg.jointTCW), '-c', '--BiLap', str(cfg.jointBiLap)]

        if cfg.externalTLWeight is not None:
            tpcCmd.extend(['--externalTLWeightFile', str(cfg.externalTLWeight)])

        print(*tpcCmd)
        subprocess.call(tpcCmd)

    makeBatchFileFromFolder(paramJsonFolder, 'txt', paramJsonFile)

    if cfg.visualizeTCFit:
        # fittingToVtk(fittingTCFolder, visualizeFittingError=True)
        targetFiles = glob.glob(join(targetVisFolder, '*.ply'))
        deformedCompleteMeshFolder = join(fittingTCFolder, 'CompleteModel')
        if cfg.numPtsCompleteMesh == 1626:
            subprocess.call(['SkeletonDeformation1626', paramJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolder])
        else:
            print('SkeletonDeformation1692', paramJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolder)
            subprocess.call(
                ['SkeletonDeformation1692', paramJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolder])

        tcVisFolder = join(visFolder, 'PureLBSTCFit')

        fittingToVtk(deformedCompleteMeshFolder, outVTKFolder=tcVisFolder, visualizeFittingError=False, addABeforeName=True, addGround=cfg.addGround,
                     meshWithFaces=inOriginalRestPoseQuadMesh)
        if cfg.visualizeCorrs:
            visualizeCorrs(targetFiles, fittingTCFolder, tcVisFolder, sanityCheck=False)

    # Run Map target pose back to rest pose
    print("Run Map target pose back to rest pose")

    if cfg.mapToRestPose:
        subprocess.call(
            ['MapTargetPtsBackToRestPose', inChunkFile, paramJsonFile, restPoseTargetFolder, '-s', inSkelDataFile,
             '-c'])

    if cfg.visualizeRestposeTargets:
        restposeTargetVisFolder = join(visFolder, 'RestposeTargets')
        os.makedirs(restposeTargetVisFolder, exist_ok=True)
        obj2vtkFolder(restPoseTargetFolder, outVtkFolder=restposeTargetVisFolder, addFaces=False)

    interpolationFolder = restPoseTargetFolder + r'\InterpolationDisplacement'
    if cfg.detailRecover:
        # Interpolate the rest pose displacement
        print("Interpolate the rest pose displacement")

        if cfg.usePoseBlendShape:
            skelData = json.load(open(inSkelDataFileComplete))
            poseBlendShape = np.array(skelData['PoseBlendShapes'])
            quanternions, trans, files = readBatchedSkelParams(paramJsonFile)
            deformBatch(paramJsonFile, fittingPoseBSFolder, inSkelDataFileComplete)
        else:
            poseBlendShape = None
            quanternions = None

        interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh(restPoseTargetFolder,
                                                                                    inSkelDataFile,
                                                                                    inOriginalRestPoseMesh,
                                                                                    interpolationFolder,
                                                                                    tw=cfg.tw,
                                                                                    interpolationSegLength=cfg.interpolationSegLength,
                                                                                    interpolationOverlappingLength=cfg.interpolationOverlappingLength,
                                                                                    chunkedInput=False,
                                                                                    spatialLap=cfg.spatialLap,
                                                                                    poseBlendShape=poseBlendShape,
                                                                                    quaternions=quanternions,
                                                                                    spatialBiLap=cfg.spatialBiLap,
                                                                                    spatialLapFromSkelData=cfg.spatialLapFromSkelData
                                                                                    )

    deformedBatchFile = prepareRestPoseDeformatoinBatch(interpolationFolder, interpolationFolder + r'\Batch')

    # Map the deformed rest pose back
    print("Map the deformed rest pose back")

    if cfg.numPtsCompleteMesh == 1626:
        subprocess.call(
            ['RecoverFleshDeformations1626', deformedBatchFile, paramJsonFile, deformedMeshFolder, '-s',
             inSkelDataFileComplete])
    elif cfg.numPtsCompleteMesh == 1692:
        print(*['RecoverFleshDeformations1692', deformedBatchFile, paramJsonFile, deformedMeshFolder, '-s',
             inSkelDataFileComplete])
        subprocess.call(
            ['RecoverFleshDeformations1692', deformedBatchFile, paramJsonFile, deformedMeshFolder, '-s',
             inSkelDataFileComplete])
    else:
        print("Unsupported number of pts:", cfg.numPtsCompleteMesh )

    if cfg.visualizeDisplacementInterpo:
        # histogram = np.copy(histogram)
        # histogram.resize((1626,))
        interpolationVisFolder = join(visFolder, 'InterpolationDisplacement')
        fittingToVtk(interpolationFolder, outVTKFolder=interpolationVisFolder, meshWithFaces=inOriginalRestPoseQuadMesh,
                     outExtName='ply', addABeforeName=True, addGround=cfg.addGround,
                     # observeHistograms=histogram,
                     visualizeFittingError=False)

    finalVisFolder = join(visFolder, 'Final')
    fittingToVtk(deformedMeshFolder, outVTKFolder=finalVisFolder, visualizeFittingError=False, addABeforeName=True,
                 meshWithFaces=inOriginalRestPoseQuadMesh, outExtName='ply', addGround=cfg.addGround)

# if __name__ == "__main__":
#     # inChunkFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\ChunkFile_Karate_7100_7700.json'
#     # outlierRemoveChunkFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\ChunkFile_Karate_7100_7700_outlierRemove.json'
#     # outputDataFolder = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\Karate_7100_7700'
#     # # fitName = '1_RegularLaplacianOnly'
#     # # fitName = '2_SpacialLaplacianOnly'
#     #
#     # # skel data used to fit to point cloud, to generate joint pose
#     # inSkelDataFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\OptRestPose\Expriment4\Fit029\Model\Skel0059.json'
#     # # skel data used to map back the restpose details
#     # inSkelDataFileComplete = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\00_SkelDataManuallyComplete.json'
#
#     inChunkFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\ChunkFile_All_6400_11500.json'
#     # outlierRemoveChunkFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\ChunkFile_All_6400_11500_outlierRemove.json'
#     outputDataFolder = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\SelectedSequences\All_6400_11500'
#     # fitName = '1_RegularLaplacianOnly'
#     # fitName = '2_SpacialLaplacianOnly'
#
#     # skel data used to fit to point cloud, to generate joint pose
#     inSkelDataFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\OptRestPose\Expriment4\Fit029\Model\Skel0059.json'
#     # skel data used to map back the restpose details
#     inSkelDataFileComplete = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\00_SkelDataManuallyComplete.json'
#
#
#
#     # outputDataFolder = join(inChunkFile, 'Interpolation3CompleteMesh')
#     # outputDataFolder = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CorrsDist\Triangulation\Interpolation3CompleteMesh'
#
#     inOriginalRestPoseMesh = r'Z:\2019_12_27_FinalLadaMesh\FinalMesh2_OnlyQuad\Mesh2_OnlyQua_No_Hole2_Tri.obj'
#     inOriginalRestPoseQuadMesh = r'Z:\2019_12_27_FinalLadaMesh\FinalMesh2_OnlyQuad\Mesh2_OnlyQua_No_Hole2.obj'
#
#
#     cfg = Config()
#     cfg.robustifierThreshold = 30
#     cfg.spatialLap = True
#     # cfg.spatialLap = False
#     cfg.tw = 1
#     cfg.interpolationSegLength = 200
#     cfg.interpolationOverlappingLength = 100
#     cfg.jointTCW = 0.1
#     cfg.jointBiLap = 0
#
#     # cfg.outliers = [
#     #     {"Ids": [1084, 1083, 1060, 1059], 'Frames':list(range(200,400))},
#     #     {"Ids": [1109, 1086], 'Frames':list(range(338, 342))},
#     #     {"Ids": [1026], 'Frames': list(range(335, 345))},
#     #     {"Ids": [1078], 'Frames': list(range(375, 385))},
#     #     {"Ids": [507], 'Frames': list(range(0, 100))},
#     # ]
#
#     cfg.outliers = [464,465,466, 1084, 1083, 1060, 1059, 1109, 1086, 1026, 1078, 507, 1085,1108]
#     cfg.outliersFrames = [
#         []
#     ]
#
#     fitName = '5_Spacial_RegularT_' + str(cfg.tw) + "_JointTW_" + str(cfg.jointTCW) + '_JBiLap_' + str(cfg.jointBiLap)  \
#               +'_Step' + str(cfg.interpolationSegLength)  + '_Overlap' + str(cfg.interpolationOverlappingLength)
#     if cfg.removeOutliers:
#         fitName = fitName + '_' + 'RemoveOutliers'
#
#     # cfg.interpolationSegLength = 200
#     # cfg.interpolationOverlappingLength = 100
#
#
#     # cfg.fitInterval = [8700, 13500]
#     cfg.fitInterval = []
#
#     cfg.doInitialFit = True
#     # cfg.doInitialFit = False
#
#     cfg.optimizeJointAngleTemporalCoherence = True
#     # cfg.optimizeJointAngleTemporalCoherence = False
#
#     cfg.mapToRestPose = True
#     # cfg.mapToRestPose = False
#     #
#     # cfg.visualizeRestposeTargets = False
#
#     cfg.detailRecover = True
#     # cfg.detailRecover = False
#
#     # cfg.visualizeTargets = False
#     #
#     # cfg.visualizeInitialFit = False
#     #
#     # cfg.visualizeTCFit = False
#     cfg.visualizeTCFit = True
#
#     cfg.removeOutliers = False
#
#     if cfg.removeOutliers:
#         inChunkFile = removeOutliers(inChunkFile, outlierRemoveChunkFile, cfg.outliers)
#
#     outputDataFolder = join(outputDataFolder, fitName)
#     fittingDataFolder = outputDataFolder + r'\Init'
#     fittingTCFolder = outputDataFolder + r'\WithTC'
#     restPoseTargetFolder = outputDataFolder + r'\RestPoseTarget'
#     deformedMeshFolder = outputDataFolder + r'\Deformed'
#     visFolder = join(outputDataFolder, 'Vis')
#
#     os.makedirs(outputDataFolder, exist_ok=True)
#     os.makedirs(fittingDataFolder, exist_ok=True)
#     os.makedirs(fittingTCFolder, exist_ok=True)
#     os.makedirs(restPoseTargetFolder, exist_ok=True)
#     os.makedirs(deformedMeshFolder, exist_ok=True)
#
#     outCfgFile = join(outputDataFolder, 'Config.json')
#     json.dump(cfg.__dict__, open(outCfgFile, 'w'), indent=2)
#
#     targetVisFolder = join(visFolder, 'Targets')
#     if cfg.visualizeTargets:
#         unpackChunkData(inChunkFile, targetVisFolder, outputType='ply')
#
#     captureData = json.load(open(inChunkFile))
#     pts = np.array(captureData['Pts'])
#     histogram = SampleDataSatistics(pts)
#
#     # Do the fit on the motion sequence
#     print("Do the fit on the motion sequence")
#     if cfg.doInitialFit:
#         if len(cfg.fitInterval) == 2:
#             fitCmd = ["CornersFitReducedQuaternionFToF1487V2", inChunkFile, fittingDataFolder,'-b', str(cfg.fitInterval[0]), '-e',  str(cfg.fitInterval[1]),
#                              '-s', inSkelDataFile, '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '20', '-c']
#         else:
#             fitCmd = ["CornersFitReducedQuaternionFToF1487V2", inChunkFile, fittingDataFolder, '-s', inSkelDataFile, '-r', str(cfg.robustifierThreshold), '--outputSkipStep', '20', '-c']
#
#         print(*fitCmd)
#         subprocess.call(fitCmd)
#
#
#     if cfg.visualizeInitialFit:
#         print("Visualize Initial Fit")
#         initialFitVTKFolder = join(visFolder, 'InitialFit')
#         fittingToVtk(fittingDataFolder, outVTKFolder=initialFitVTKFolder)
#         # visualizeCorrs(targetFiles, fittingDataFolder, fittingDataFolder + r'\Corrs', sanityCheck=False)
#
#     # make the batch file for fitting parameters and targets
#     print("Make the batch file for fitting parameters")
#     fittingParamFolder = fittingDataFolder + r"\Params"
#
#     initialParamJsonFile = fittingParamFolder + "\\0Poses.json"
#     makeBatchFileFromFolder(fittingParamFolder, 'txt', initialParamJsonFile)
#
#     # Run temporal coherence optimization
#
#     paramJsonFolder = fittingTCFolder + "\\Params"
#     paramJsonFile = paramJsonFolder + "\\0Poses.json"
#     if cfg.optimizeJointAngleTemporalCoherence:
#         print("Run temporal coherence optimization")
#         tpcCmd =  ['CornersFitReducedBatchedQuaternionTCWithInit', inChunkFile, initialParamJsonFile, fittingTCFolder, '-s',
#              inSkelDataFile, '-j', str(cfg.jointTCW), '-c', '--BiLap', str(cfg.jointBiLap)]
#         print(*tpcCmd)
#         subprocess.call(tpcCmd)
#         makeBatchFileFromFolder(paramJsonFolder, 'txt', paramJsonFile)
#
#     if cfg.visualizeTCFit:
#         # fittingToVtk(fittingTCFolder, visualizeFittingError=True)
#         targetFiles = glob.glob(join(targetVisFolder, '*.ply'))
#         deformedCompleteMeshFolder = join(fittingTCFolder, 'CompleteModel')
#         subprocess.call(['SkeletonDeformation1626', paramJsonFile, inSkelDataFileComplete, deformedCompleteMeshFolder])
#         tcVisFolder = join(visFolder, 'TCFit')
#
#         visualizeCorrs(targetFiles, fittingTCFolder, tcVisFolder, sanityCheck=False)
#
#         fittingToVtk(deformedCompleteMeshFolder, outVTKFolder=tcVisFolder, visualizeFittingError=False, meshWithFaces=inOriginalRestPoseQuadMesh)
#
#
#     # Run Map target pose back to rest pose
#     print("Run Map target pose back to rest pose")
#
#     if cfg.mapToRestPose:
#             subprocess.call(
#                 ['MapTargetPtsBackToRestPose', inChunkFile, paramJsonFile, restPoseTargetFolder, '-s', inSkelDataFile, '-c'])
#
#     if cfg.visualizeRestposeTargets:
#         restposeTargetVisFolder = join(visFolder, 'RestposeTargets')
#         os.makedirs(restposeTargetVisFolder, exist_ok=True)
#         obj2vtkFolder(restPoseTargetFolder, outVtkFolder=restposeTargetVisFolder, addFaces=False)
#
#     interpolationFolder = restPoseTargetFolder + r'\InterpolationDisplacement'
#     if cfg.detailRecover:
#             # Interpolate the rest pose displacement
#         print("Interpolate the rest pose displacement")
#
#         interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh(restPoseTargetFolder, inSkelDataFile, inOriginalRestPoseMesh, interpolationFolder,
#                                 tw=cfg.tw, interpolationSegLength= cfg.interpolationSegLength, interpolationOverlappingLength=cfg.interpolationOverlappingLength, chunkedInput=False, spatialLap=cfg.spatialLap)
#         deformedBatchFile = prepareRestPoseDeformatoinBatch(interpolationFolder, interpolationFolder + r'\Batch')
#
#
#         # Map the deformed rest pose back
#         print("Map the deformed rest pose back")
#
#         if cfg.numPtsCompleteMesh == 1626:
#             subprocess.call(
#                 ['RecoverFleshDeformations1626', deformedBatchFile, paramJsonFile, deformedMeshFolder, '-s',
#                  inSkelDataFileComplete])
#         elif cfg.numPtsCompleteMesh == 1692:
#             subprocess.call(
#                 ['RecoverFleshDeformations1692', deformedBatchFile, paramJsonFile, deformedMeshFolder, '-s',
#                  inSkelDataFileComplete])
#         else:
#             print("Unsupported number of pts:", cfg.numPtsCompleteMesh )
#
#     if cfg.visualizeDisplacementInterpo:
#         # histogram = np.copy(histogram)
#         # histogram.resize((1626,))
#         interpolationVisFolder = join(visFolder, 'InterpolationDisplacement')
#         fittingToVtk(interpolationFolder, outVTKFolder=interpolationVisFolder, meshWithFaces=inOriginalRestPoseQuadMesh, extName='ply',
#                      # observeHistograms=histogram,
#                      visualizeFittingError=False)
#
#     finalVisFolder = join(visFolder, 'Final')
#     fittingToVtk(deformedMeshFolder, outVTKFolder=finalVisFolder, visualizeFittingError=False, meshWithFaces=inOriginalRestPoseQuadMesh)
#     # visualizeCorrs(targetFiles, deformedMeshFolder, deformedMeshFolder + r'\Corrs', sanityCheck=False)
