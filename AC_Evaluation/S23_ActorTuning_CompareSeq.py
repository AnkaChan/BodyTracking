from Utility import *
import pyvista as pv
from M01_ARAPDeformation import *
from SkelFit.Data import *
from M02_ObjConverter import removeVertsFromMeshFolder

def visualizeFitting(inFittingFolder, outFolder, ARAP=False, completeMeshFile=None, corrs=None, ext='ply'):
    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIdsWithNeck.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'

    exampleQuadMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'

    finalFolder = inFinalModelFitFolder

    badRightFootVIdsFile = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\Data\S_22Actorcalibration\BadRightFeetIds\RFeet.json'
    rightFootVIds = json.load(open(badRightFootVIdsFile))

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))
    isolatedVerts = getIsolatedVerts(pv.PolyData(exampleQuadMesh))
    vertsToRemove = set(headVIds + handVIds + isolatedVerts.tolist())

    inFiles = sortedGlob(join(inFittingFolder, '*.'+ext))
    os.makedirs(outFolder, exist_ok=True)
    if ARAP:
        outARAPDeformFolder = join(outFolder, 'ARAP_obj')
        os.makedirs(outARAPDeformFolder, exist_ok=True)
        # for inFile in tqdm.tqdm(inFiles, desc='ARAP deformation'):
        #     mesh = pv.PolyData(inFile)
        #     meshFinal = pv.PolyData(join(finalFolder, Path(inFile).stem + '.ply'))
        #     for iV in range(mesh.points.shape[0]):
        #         if iV in rightFootVIds:
        #             mesh.points[iV, :] = meshFinal.points[iV, :]
        #
        #     write_obj('Cache.obj', mesh.points)
        #     outObjMeshFile = join(outARAPDeformFolder, Path(inFile).stem + '.obj')
        #     # ARAPDeformation(completeMeshFile, 'Cache.obj', outObjMeshFile, corrs)

        removeVertsFromMeshFolder(outARAPDeformFolder, outFolder,
                                  vertsToRemove, exampleQuadMesh, inExtName='obj', removeVerts=False,)
    else:
        removeVertsFromMeshFolder(inFittingFolder, outFolder,
                                  vertsToRemove, exampleQuadMesh, inExtName='ply', removeVerts=False,)

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

        dis = np.sqrt(diff[:, 0]**2+diff[:, 1]**2+diff[:, 2]**2)

        dis[np.where(targetPC.points[:, 2]==-1)[0]] = -1
        dis[targetPC.points.shape[0]:] = -1

        json.dump(dis.tolist(), open(join(outFolder, Path(inMesh).stem+'.json'), 'w'))

        loop.set_description("AvgErr:" + str(np.mean(dis[np.where(targetPC.points[:, 2]!=-1)[0]])))

if __name__ == '__main__':
    inInitialModelFitFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand_Initial_SkelModel\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init'
    inInitialModelMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\014_SkelDataKateyXPose.json.vtk'
    inFinalModelFitFolder = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init'
    completeObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete_HandEdited.obj'
    # completeObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri.obj'
    outFolder = join(r'F:\WorkingCopy2\2021_01_09_ActorTuningVis', 'SeqComparison', 'Katey_Stand')
    triangulationFolder = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose\Triangulation_RThres1.5_HardRThres_1.5'

    frameNames = ['A' + str(i).zfill(8) + '.obj' for i in range(3500, 5000)]

    outInitialModelFitFolder = join(outFolder, 'Initial')
    outFinalModelFitFolder = join(outFolder, 'Final')

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
        # elif iV in rightFootVIds:
        #     corrs.append([int(iV), iInvalidTarget])


    print("Number of constraints: ", len(corrs))

    # visualizeFitting(inInitialModelFitFolder, outInitialModelFitFolder, ARAP=True, completeMeshFile=completeObjMeshFile, corrs=corrs)
    #
    # visualizeFitting(inFinalModelFitFolder, outFinalModelFitFolder, ARAP=False)

    computeFitingErrs(outInitialModelFitFolder, triangulationFolder, frameNames, join(outInitialModelFitFolder, 'Errs'))
    computeFitingErrs(outFinalModelFitFolder, triangulationFolder, frameNames, join(outFinalModelFitFolder, 'Errs'))
    # compute fiting errors
