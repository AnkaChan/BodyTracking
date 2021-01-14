from S02_RecoverDetails import *
from shutil import copy
import pyvista as pv
import tqdm
from SkelFit.Data import *
from SkelFit.Visualization import *
from M_ARAPDeformation import *


def computeFitingErrs(inDeformedMeshFolder, triangulationFolder, frameNames, outFolder):
    os.makedirs(outFolder, exist_ok=True)

    inMeshes = sortedGlob(join(inDeformedMeshFolder, '*.ply'))
    errs = []
    loop = tqdm.tqdm(zip(inMeshes, frameNames))
    for inMesh, frameName in loop:
        deformedMesh = pv.PolyData(inMesh)
        targetPC = pv.PolyData(join(triangulationFolder, frameName))

        diff = np.zeros(deformedMesh.points.shape)
        diff[:targetPC.points.shape[0], :] = deformedMesh.points[:targetPC.points.shape[0], :] - targetPC.points

        dis = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)

        dis[np.where(targetPC.points[:, 2] == -1)[0]] = -1
        dis[targetPC.points.shape[0]:] = -1

        json.dump(dis.tolist(), open(join(outFolder, Path(inMesh).stem + '.json'), 'w'))

        loop.set_description("AvgErr:" + str(np.mean(dis[np.where(targetPC.points[:, 2] != -1)[0]])))

if __name__ == '__main__':
    inSKelFoler = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\Data\allSkelDatas\SkelDatas_Katey'

    inChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Stand_1Frame_1353_1354.json'
    outputFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Transition'
    transitionAnimationFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Transition\Transition'
    outFileName = r'A00004799.ply'
    # skelDataFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\S01_Combined_Katey_HandHead_OriginalRestpose.json'
    # skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\014_SkelDataKateyXPose.json' # Katey initial skel data file
    completeObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete_HandEdited.obj'

    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'
    exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'
    triangulationFolder = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose\Triangulation_RThres1.5_HardRThres_1.5'

    cfgSkelFit = M01_LBSFitting.Config()
    cfgSkelFit.poseChangeRegularizerWeight = 1000
    cfgSkelFit.inOriginalRestPoseMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete.obj'
    # cfgSkelFit.mapToRestPose = False
    cfgSkelFit.interpolationSegLength = 1
    cfgSkelFit.interpolationOverlappingLength = 0

    allSkelDatas = sortedGlob(join(inSKelFoler, '*.json'))
    os.makedirs(transitionAnimationFolder, exist_ok=True)

    transitionAnimFittedFolder = join(transitionAnimationFolder, 'Fitted')
    transitionAnimArapFolder = join(transitionAnimationFolder, 'ARAP_Obj')
    transitionAnimFinalFolder = join(transitionAnimationFolder, 'Final')
    os.makedirs(transitionAnimFittedFolder, exist_ok=True)
    os.makedirs(transitionAnimArapFolder, exist_ok=True)
    os.makedirs(transitionAnimFinalFolder, exist_ok=True)

    for i, skelDataFile in enumerate(allSkelDatas):
        outputFolderSkelData = join(outputFolder, str(i).zfill(2))
        # M01_LBSFitting.lbsFitting(inChunkedFile, outputFolderSkelData, skelDataFile, cfgSkelFit)
        outFolderName = M01_LBSFitting.getFitName(cfgSkelFit)
        outputFile = join(outputFolderSkelData, outFolderName, 'Init', outFileName)
        mesh = pv.PolyData(outputFile)
        outCopiedFile = join(transitionAnimFittedFolder, Path(outFileName).stem + '_' + str(i).zfill(2)  + '.obj')
        # copy(outputFile, outCopiedFile, )
        write_obj(outCopiedFile, mesh.points)

    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'
    inInitialModelMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\014_SkelDataKateyXPose.json.vtk'

    exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'

    numRealPts = 1487
    targetMesh = pv.PolyData(inInitialModelMeshFile)
    corrs = []

    # for isolated points in first numRealPts

    iInvalidTarget = -1
    for iTargetV in range(numRealPts):
        if targetMesh.points[iTargetV, 2] == -1:
            iInvalidTarget = iTargetV
        # for bad verts we also need to constrain it
        # if iTargetV not in rightFootVIds:
        #     corrs.append([iTargetV, iTargetV])

        corrs.append([iTargetV, iTargetV])

    isolatedVerts = getIsolatedVerts(pv.PolyData(completeObjMeshFile))
    for iV in isolatedVerts:
        if iV >= numRealPts:
            corrs.append([int(iV), iInvalidTarget])

    badRightFootVIdsFile = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\Data\S_22Actorcalibration\BadRightFeetIds\RFeet.json'
    rightFootVIds = json.load(open(badRightFootVIdsFile))

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))
    isolatedVerts = getIsolatedVerts(pv.PolyData(exampleQuadMesh))
    vertsToRemove = set(headVIds + handVIds + isolatedVerts.tolist())

    inFiles = sortedGlob(join(transitionAnimFittedFolder, '*.obj'))
    for i, inFile in tqdm.tqdm(enumerate(inFiles), desc='ARAP deformation'):
        mesh = pv.PolyData(inFile)
        if i == 0:
            meshFinal = pv.PolyData(inFiles[2])
            for iV in range(mesh.points.shape[0]):
                if iV in rightFootVIds:
                    mesh.points[iV, :] = meshFinal.points[iV, :]

        write_obj('Cache.obj', mesh.points)
        outObjMeshFile = join(transitionAnimArapFolder, Path(inFile).stem + '.obj')
    #     ARAPDeformation(completeObjMeshFile, 'Cache.obj', outObjMeshFile, corrs)
    #
    # removeVertsFromMeshFolder(transitionAnimArapFolder, transitionAnimFinalFolder,
    #                          vertsToRemove, exampleQuadMesh, inExtName='obj')

    frameNames = [Path(outFileName).stem + '.obj' for i in inFiles]

    computeFitingErrs(transitionAnimFinalFolder, triangulationFolder, frameNames, join(transitionAnimFinalFolder, 'Errs'))
