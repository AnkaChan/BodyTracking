import M01_LBSFitting
from M02_ObjConverter import removeVertsFromMeshFolder
import M02_ObjConverter
from Utility import *
import json
from SkelFit.Data import getIsolatedVerts
from SkelFit.Visualization import fittingToVtk, unpackChunkData
from M03_TemporalSmoothing import temporalSmoothingPointTrajectory
if __name__ == '__main__':
    cfgSkelFit = M01_LBSFitting.Config()

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Ground.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Lada_Ground'

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Stand.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Lada_Stand'
    #
    # skelDataFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\S02_Combined_Lada_HandHead_OriginalRestpose.json'

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Stand_3500_5000.json'
    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Stand_54_1554.json'
    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_CalibrationSeqs_54_6054_outlierFiltered.json'

    # inChunkedFile = r'F:\WorkingCopy2\2020_01_13_FinalAnimations\Katey_NewPipeline\LongSequence\LongSequence_0_2800.clean.json' # for testing the optimized model
    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_LongSeq.json' # for testing the optimized model
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_LongSeq'

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Rolling_3440_3762.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Rolling'

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Ballet_5422_5423.json' # Katey_Ballet_5422_5423
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Ballet'
    #
    inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Interpolation_ReduceFlickering.json' # Katey_Ballet_5422_5423
    outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Interpolation'


    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_SquatSpin.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_SquatSpin'

    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Yoga_9613_10142.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Lada_Yoga'

    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand_Initial_SkelModel'
    # skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\016_SkelDataKateyInitialFromSmpl.json' # Katey initial skel data file
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Initial'
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Initial' # for testing the intial model

    skelDataFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\S01_Combined_Katey_HandHead_OriginalRestpose.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Final'
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Final' # for testing the optimized model


    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'
    exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'
    # exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\CompleteBetterFeet_tri_backToRestpose.obj'

    # inChunkedFile = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\Inpainted\Threshold_0inpaint_3_MaxMovement15\ChunkFile.json'
    #
    # skelDataFile = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Modelling2\01_SkelData_Marianne_Complete_newMesh.json'
    # cfgSkelFit.omittedVertsFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh_Marianne\HandVertsMarianne.json'

    # exampleQuadMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2.obj'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Marianne'

    cfgSkelFit.inOriginalRestPoseMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri.obj'
    # cfgSkelFit.inOriginalRestPoseMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2_no_hole_tri_clear.obj'
    # cfgSkelFit.mapToRestPose = False

    # for fitting to existing seqs
    # cfgSkelFit.interpolationSegLength = 1
    # cfgSkelFit.interpolationOverlappingLength = 0

    cfgSkelFit.poseChangeRegularizerWeight = 600
    cfgSkelFit.jointTCW = 200

    cfgSkelFit.interpolationSegLength = 150
    cfgSkelFit.interpolationOverlappingLength = 50
    # cfgSkelFit.interpolationSegLength = 500
    # cfgSkelFit.interpolationOverlappingLength = 100
    # cfgSkelFit.interpolationSegLength = 322 # for Katey rolling
    # cfgSkelFit.interpolationSegLength = 299 # for Katey rolling
    # cfgSkelFit.interpolationOverlappingLength = 0
    # cfgSkelFit.poseChangeRegularizerWeight = 500
    # cfgSkelFit.poseChangeRegularizerWeight = 10 # for katey's long seq
    # cfgSkelFit.poseChangeRegularizerWeight = 400 # for katey's rolling
    # cfgSkelFit.poseChangeRegularizerWeight = 800 # for katey's squat spin
    cfgSkelFit.tw = 100
    cfgSkelFit.removeOutliers =True
    # cfgSkelFit.removeOutliers =False
    cfgSkelFit.outlierFilterThreshold = 50
    # cfgSkelFit.smoothingSoftConstraintWeight = 20
    cfgSkelFit.smoothingSoftConstraintWeight = 1
    # cfgSkelFit.detailRecover = False

    # M01_LBSFitting.lbsFitting(inChunkedFile, outputFolder, skelDataFile, cfgSkelFit)

    outputFolder = join(outputFolder, M01_LBSFitting.getFitName(cfgSkelFit))
    # paramFile = join(outputFolder, 'Init', 'Params', 'Params.json')
    paramFile = join(outputFolder, 'LBSWithTC', 'Params', 'Params.json')
    if cfgSkelFit.removeOutliers:
        inChunkedFile = inChunkedFile + '.cleaned.json'
    # M01_LBSFitting.detailInterpolation(inChunkedFile, outputFolder, skelDataFile, paramFile, cfgSkelFit)

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))
    vertsToRemove = set(headVIds + handVIds)
    # isolatedPoints = getIsolatedVerts(cfgSkelFit.inOriginalRestPoseMesh)
    # vertsToRemove = set(headVIds + handVIds + isolatedPoints)

    # unpackChunkData(inChunkedFile, join(outputFolder, 'Target'), outputType='ply')


    # temporalSmoothingPointTrajectory(join(outputFolder, 'Interpolated'), join(outputFolder, 'Smoothed'), softConstraintWeight=cfgSkelFit.smoothingSoftConstraintWeight)
    # removeVertsFromMeshFolder(join(outputFolder, 'Smoothed'), join(outputFolder, 'clean'), vertsToRemove, exampleQuadMesh, removeVerts=False)
    removeVertsFromMeshFolder(join(outputFolder, 'Interpolated'), join(outputFolder, 'clean'), vertsToRemove, exampleQuadMesh, removeVerts=False)
    # removeVertsFromMeshFolder(join(outputFolder, 'LBSWithTC'), join(outputFolder, 'clean'), vertsToRemove, exampleQuadMesh, removeVerts=False)

    # removeVertsFromMeshFolder(r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey', r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\Cleaned'
    #                           , vertsToRemove, exampleQuadMesh, removeVerts=False, inExtName='obj')

    # temporalSmoothingPointTrajectory(join(outputFolder, 'Interpolated'), join(outputFolder, 'Smoothed'))
    # fittingToVtk(join(outputFolder, 'Smoothed'), outVTKFolder=join(outputFolder, 'Smoothed', 'clean'), meshWithFaces=exampleQuadMesh, removeUnobservedFaces=False, extName='ply', outExtName='ply')