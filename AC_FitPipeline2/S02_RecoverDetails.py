import M01_LBSFitting
from M02_ObjConverter import removeVertsFromMeshFolder
import M02_ObjConverter
from Utility import *
import json
from SkelFit.Data import getIsolatedVerts

if __name__ == '__main__':
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

    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand_Initial_SkelModel'
    # skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\016_SkelDataKateyInitialFromSmpl.json' # Katey initial skel data file
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Initial'
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Initial' # for testing the intial model

    # skelDataFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\S01_Combined_Katey_HandHead_OriginalRestpose.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Training_Final'
    # # outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Evaluation\Test_Final' # for testing the optimized model
    #
    #
    # headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    # handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'
    # exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'

    inChunkedFile = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\Inpainted\Threshold_0inpaint_3_MaxMovement15\ChunkFile.json'

    skelDataFile = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Modelling2\01_SkelData_Marianne_Complete_newMesh.json'
    inOriginalRestPoseMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2_no_hole_tri_clear.obj'
    inOriginalRestPoseQuadMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2.obj'
    outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Marianne'

    cfgSkelFit = M01_LBSFitting.Config()
    # cfgSkelFit.inOriginalRestPoseMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete.obj'
    cfgSkelFit.inOriginalRestPoseMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2_no_hole_tri_clear.obj'
    cfgSkelFit.mapToRestPose = False
    cfgSkelFit.interpolationSegLength = 400
    cfgSkelFit.interpolationOverlappingLength = 200
    cfgSkelFit.poseChangeRegularizerWeight = 2000
    cfgSkelFit.tw = 1000
    # cfgSkelFit.detailRecover = False
    cfgSkelFit.omittedVertsFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh_Marianne\HandVertsMarianne.json'

    # M01_LBSFitting.lbsFitting(inChunkedFile, outputFolder, skelDataFile, cfgSkelFit)

    outputFolder = join(outputFolder, M01_LBSFitting.getFitName(cfgSkelFit))
    paramFile = join(outputFolder, 'Init', 'Params', 'Params.json')
    # paramFile = join(outputFolder, 'LBSWithTC', 'Params', 'Params.json')
    M01_LBSFitting.detailInterpolation(inChunkedFile, outputFolder, skelDataFile, paramFile, cfgSkelFit)

    # headVIds = json.load(open(headVIdsFile))
    # handVIds = json.load(open(handVIdsFile))
    # isolatedPoints = getIsolatedVerts(cfgSkelFit.inOriginalRestPoseMesh)
    # vertsToRemove = set(headVIds + handVIds + isolatedPoints)

    # removeVertsFromMeshFolder(join(outputFolder, 'Interpolated'), join(outputFolder, 'Interpolated', 'clean'), vertsToRemove, exampleQuadMesh)