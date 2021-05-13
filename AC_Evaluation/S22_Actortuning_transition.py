import pyvista as pv
from Utility import *
from shutil import copy
from SkelFit.SkeletonModel import *
from SkelFit.Data  import *
from S23_ActorTuning_CompareSeq import visualizeFitting
from S23_ActorTuning_CompareSeq import visualizeErrs


def computeFitingErrs(inDeformedMeshFolder, triangulationFile):
    outFolder = join(inDeformedMeshFolder, 'errs')
    os.makedirs(outFolder, exist_ok=True)

    inMeshes = sortedGlob(join(inDeformedMeshFolder, '*.ply'))
    errs = []
    loop = tqdm.tqdm(inMeshes)
    for inMesh in loop:
        deformedMesh = pv.PolyData(inMesh)
        targetPC = pv.PolyData(triangulationFile)

        diff = np.zeros(deformedMesh.points.shape)
        diff[:targetPC.points.shape[0], :] = deformedMesh.points[:targetPC.points.shape[0], :] - targetPC.points

        dis = np.sqrt(diff[:, 0]**2+diff[:, 1]**2+diff[:, 2]**2)

        dis[np.where(targetPC.points[:, 2]==-1)[0]] = -1
        dis[targetPC.points.shape[0]:] = -1

        json.dump(dis.tolist(), open(join(outFolder, Path(inMesh).stem+'.json'), 'w'))

        loop.set_description("AvgErr:" + str(np.mean(dis[np.where(targetPC.points[:, 2]!=-1)[0]])))

if __name__ == '__main__':
    # restposeTransitionFolder = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\Data\S_22Actorcalibration\RestposeChange'
    # inFiles = sortedGlob(join(restposeTransitionFolder, '*.vtk'))
    # restposeTransitionFolderFixedFolder = join(restposeTransitionFolder, 'Repaired')
    # os.makedirs(restposeTransitionFolderFixedFolder, exist_ok=True)
    #
    # exampleMesh = pv.PolyData(r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\014_SkelDataKateyXPose.json.vtk')
    #
    # for file in inFiles:
    #     mesh = pv.PolyData(file)
    #     mesh.faces = exampleMesh.faces
    #
    #     mesh.save(join(restposeTransitionFolderFixedFolder, Path(file).stem + '.ply'))

    actorCalibrationFolder = r'F:\WorkingCopy2\2020_01_01_KateyCapture\ActorCalibration\0'
    skelDataOutFolder = join('.', 'Data', 'allSkelDatas', 'SkelDatas_Katey')
    # skelTransitionOutFolder = join('.', 'Data', 'allSkelDatas', 'OptimizationTransition')
    skelTransitionOutFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Transition\Transition\Fitted'
    inPoseBatchFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Output\Katey_Stand\SLap_SBiLap_True_TLap_1_JTW_0.5_JBiLap_0_Step1_Overlap0\Init\Params\Params.json'
    completeObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete_HandEdited.obj'
    inInitialModelMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\CeresSkelFit\PrepareData1487\014_SkelDataKateyXPose.json.vtk'



    poseId = 1299
    os.makedirs(skelDataOutFolder, exist_ok=True)
    os.makedirs(skelTransitionOutFolder, exist_ok=True)

    calibOutputFolder = sortedGlob(join(actorCalibrationFolder, 'Fit*'))
    # print(calibOutputFolder)

    allSkelDatas = []
    # for calibFolder in calibOutputFolder:
    #     skelDataFiles = glob.glob(join(calibFolder, 'Model', '*.json'))
    #     allSkelDatas = allSkelDatas + skelDataFiles
    #
    # for skelDataFile in allSkelDatas:
    #     copy(skelDataFile, join(skelDataOutFolder, Path(skelDataFile).stem + '.json'))
    #
    # # deform and generate transition
    #
    # for i, skelDataFile in enumerate(allSkelDatas):
    #     vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(skelDataFile)
    #     qs, ts = loadPoseChunkFile(inPoseBatchFile)
    #
    #     q = qs[poseId]
    #     t = np.array(ts[poseId][0])
    #     Rs = quaternionsToRotations(q)
    #
    #     newVerts = deformed = deformVerts(vRestpose, Rs, t, J, weights, kintreeTable, parent)
    #
    #     write_obj(join(skelTransitionOutFolder, 'Transition_'+ str(i).zfill(2)+'.obj'), newVerts)

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

    # visualizeFitting(skelTransitionOutFolder, join(skelTransitionOutFolder, 'complete'), ARAP=True, completeMeshFile=completeObjMeshFile, corrs=corrs, ext='obj')
    inFolder = r'F:\WorkingCopy2\2021_01_09_ActorTuningVis\Transition\Transition\ARAP_Obj\clean'
    inTargetFile = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose\Triangulation_RThres1.5_HardRThres_1.5\vis\A00004799.ply'

    # break first and second iteration in half
    meshFiles = sortedGlob(join(inFolder, '*.ply'))

    meshes = [pv.PolyData(meshFile ) for meshFile in meshFiles]
    newMeshes1 = pv.PolyData(meshFiles[0])
    newMeshes1_2 = pv.PolyData(meshFiles[0])
    newMeshes1_3 = pv.PolyData(meshFiles[0])
    newMeshes1.points = meshes[0].points*0.75 + meshes[1].points*0.25
    newMeshes1_2.points = meshes[0].points*0.5 + meshes[1].points*0.5
    newMeshes1_3.points = meshes[0].points*0.25 + meshes[1].points*0.75

    newMeshes2 = pv.PolyData(meshFiles[0])
    newMeshes2.points = meshes[1].points*0.5 + meshes[2].points*0.5

    meshes.insert(1, newMeshes1)
    meshes.insert(2, newMeshes1_2)
    meshes.insert(3, newMeshes1_3)

    meshes.insert(5, newMeshes2)

    outFolder = join(inFolder, 'WithExtraFrames')
    os.makedirs(outFolder, exist_ok=True)
    for i, mesh in enumerate(meshes):
        mesh.save(join(outFolder, 'Mesh'+str(i).zfill(3)+'.ply'))

    computeFitingErrs(outFolder, inTargetFile, )
    visualizeErrs(outFolder, outFolder + r'\Errs')