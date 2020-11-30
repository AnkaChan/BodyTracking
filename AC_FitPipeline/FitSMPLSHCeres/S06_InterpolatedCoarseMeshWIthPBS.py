from SkelFit.Visualization import *
from SkelFit.Data import *
import os
import json
from pathlib import Path
from InterpolateRestposeFleshDeformationDisplacement import interpolateRestPoseDeformation, \
    prepareRestPoseDeformatoinBatch, interpolateRestPoseDeformationWithTemporalCoherence3RegularLapDifferentMesh
from SkelFit.PoseBlendShape import *
from SkelFit.PoseBlendShape import deformWithPoseBlendShapes

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


def interpolateCoarseMeshOnRestpose(inPoseFile, inTargetJsonFile, outputDataFolder, inSkelDataFile, inOriginalRestPoseMesh, inOriginalRestPoseQuadMesh, cfg = Config()):
    # fitName = getFitName(inChunkFile, outputDataFolder, inSkelDataFile, inSkelDataFileComplete, inOriginalRestPoseMesh, inOriginalRestPoseQuadMesh, cfg = Config())

    fittingPoseBSFolder = outputDataFolder + r'\LBSWithPoseBS'
    restPoseTargetFolder = outputDataFolder + r'\RestPoseTarget'
    deformedMeshFolder = outputDataFolder + r'\Deformed'
    visFolder = join(outputDataFolder, 'Vis')

    os.makedirs(outputDataFolder, exist_ok=True)
    os.makedirs(restPoseTargetFolder, exist_ok=True)
    os.makedirs(deformedMeshFolder, exist_ok=True)
    os.makedirs(fittingPoseBSFolder, exist_ok=True)

    # map target back to rest pose
    subprocess.call(
        [r'Bin\MapTargetPtsBackToRestPose', inTargetJsonFile, inPoseFile, restPoseTargetFolder, '-s', inSkelDataFile,
         '-c'])

    if cfg.visualizeRestposeTargets:
        restposeTargetVisFolder = join(visFolder, 'RestposeTargets')
        os.makedirs(restposeTargetVisFolder, exist_ok=True)
        obj2vtkFolder(restPoseTargetFolder, outVtkFolder=restposeTargetVisFolder, addFaces=False)

    interpolationFolder = restPoseTargetFolder + r'\InterpolationDisplacement'

    if cfg.usePoseBlendShape:
        skelData = json.load(open(inSkelDataFile))
        poseBlendShape = np.array(skelData['PoseBlendShapes'])
        quanternions, trans, files = readBatchedSkelParams(inPoseFile)
        deformBatch(inPoseFile, fittingPoseBSFolder, inSkelDataFile)
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





if __name__ == '__main__':
    inPoseFile = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\GroundMotionNewPipeline\SLap_SBiLap_True_TLap_50_JTW_5000_JBiLap_0_Step200_Overlap100\LBSWithTC\Params\0Poses.json'
    inTargetJsonFile = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\GroundMotionNewPipeline\OutlierFiltered\GroundMotionNewPipeline_0_2000.json'
    skelFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\09_SkelDataLadaWithPBS.json'
    inOriginalRestPoseMesh = r'F:\WorkingCopy2\2020_01_16_KM_Edited_Meshes\KateyCalibrated_edited3_tri.obj'
    inOriginalRestPoseQuadMesh = r'F:\WorkingCopy2\2020_01_16_KM_Edited_Meshes\KateyCalibrated_edited3.obj'
    outputDataFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\InterpolateWIthPBS'
    cfg = Config()

    cfg.numPtsCompleteMesh = 1692
    cfg.visualizeInitialFit = True
    cfg.spatialLap = True
    cfg.spatialBiLap = True
    cfg.meshInterpolationCfg.usePoseBlendShape = True
    # cfg.tw = 0
    cfg.meshInterpolationCfg.tw = 50
    # cfg.meshInterpolationCfg.tw = 100 # for Yoga

    # cfg.meshInterpolationCfg.interpolationSegLength = 100
    cfg.interpolationSegLength = 10
    cfg.interpolationOverlappingLength = 0

    # cfg.meshInterpolationCfg.interpolationSegLength = 200
    # cfg.meshInterpolationCfg.interpolationOverlappingLength = 100

    cfg.meshInterpolationCfg.interpolationSegLength = 500
    cfg.meshInterpolationCfg.interpolationOverlappingLength = 200

    # cfg.meshInterpolationCfg.jointTCW = 1
    cfg.meshInterpolationCfg.poseChangeRegularizerWeight = 200
    # cfg.meshInterpolationCfg.jointTCW = 10000 # for Ground Motion
    # cfg.meshInterpolationCfg.jointTCW = 2000 # for Stand Motion
    cfg.meshInterpolationCfg.jointTCW = 5000  # for Long Sequence Katey Moti

    interpolateCoarseMeshOnRestpose(inPoseFile, inTargetJsonFile, outputDataFolder, skelFile, inOriginalRestPoseMesh, inOriginalRestPoseQuadMesh, cfg)
