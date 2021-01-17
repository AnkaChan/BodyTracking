import pyvista as pv
from Utility import *
from shutil import copy
from SkelFit.SkeletonModel import *
from SkelFit.Data  import *
from S23_ActorTuning_CompareSeq import visualizeFitting

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

    visualizeFitting(skelTransitionOutFolder, join(skelTransitionOutFolder, 'complete'), ARAP=True, completeMeshFile=completeObjMeshFile, corrs=corrs, ext='obj')