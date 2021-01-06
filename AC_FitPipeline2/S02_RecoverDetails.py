import M01_LBSFitting
from M02_ObjConverter import removeVertsFromMeshFolder
import M02_ObjConverter
from Utility import *
import json

if __name__ == '__main__':
    # inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Ground.json'
    # outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Lada_Ground'

    inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Stand.json'
    outputFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Lada_Stand'

    skelDataFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\S02_Combined_Lada_HandHead_OriginalRestpose.json'

    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'
    exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'

    cfgSkelFit = M01_LBSFitting.Config()
    cfgSkelFit.inOriginalRestPoseMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri_backToRestpose.obj'
    # cfgSkelFit.mapToRestPose = False
    cfgSkelFit.interpolationSegLength = 1
    cfgSkelFit.interpolationOverlappingLength = 0
    # cfgSkelFit.detailRecover = False

    # M01_LBSFitting.lbsFitting(inChunkedFile, outputFolder, skelDataFile, cfgSkelFit)

    outputFolder = join(outputFolder, M01_LBSFitting.getFitName(cfgSkelFit))
    paramFile = join(outputFolder, 'Init', 'Params', 'Params.json')
    # paramFile = join(outputFolder, 'LBSWithTC', 'Params', 'Params.json')
    # M01_LBSFitting.detailInterpolation(inChunkedFile, outputFolder, skelDataFile, paramFile, cfgSkelFit)

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))
    vertsToRemove = headVIds + handVIds

    removeVertsFromMeshFolder(join(outputFolder, 'Interpolated'), join(outputFolder, 'Interpolated', 'clean'), vertsToRemove, exampleQuadMesh)