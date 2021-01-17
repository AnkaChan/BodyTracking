from S04_FigureOutTheDeformationModel_ import *
from S07_MapEditedMeshBackToRestpose import *
import pyigl as igl
from iglhelpers import *
import json

if __name__ == '__main__':
    inTargetObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh.obj'
    inSourceObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri.obj'
    # outObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete.obj'
    outObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\KateyRestposeMesh_Complete_HandEdited.obj'
    inCoarseSkelData = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'
    inCoarseJoints = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json.Bone.vtk'

    inJsonSkelData = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\S02_Combined_Lada_HandHead_OriginalRestpose.json'
    outputNewSkelFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\S01_Combined_Katey_HandHead_OriginalRestpose.json'

    jsonSkelData = json.load(open(inJsonSkelData))
    jsonSkelDataCoarse = json.load(open(inCoarseSkelData))

    # ARAP deformation map Lada's restpose mesh to Katey's
    numCoarseJoints = 16
    numRealPts = 1487
    targetMesh = pv.PolyData(inTargetObjMeshFile)
    corrs = []

    # for isolated points in first numRealPts
    iInvalidTarget = -1
    for iTargetV in range(numRealPts):
        if targetMesh.points[iTargetV,2] == -1:
            iInvalidTarget = iTargetV
        # for bad verts we also need to constrain it
        corrs.append([iTargetV, iTargetV])

    isolatedVerts = getIsolatedVerts(pv.PolyData(inSourceObjMeshFile))
    for iV in isolatedVerts:
        if iV >= numRealPts:
            corrs.append([int(iV), iInvalidTarget])

    print("Number of constraints: ", len(corrs))

    # ARAPDeformation(inSourceObjMeshFile, inTargetObjMeshFile, outObjMeshFile, corrs)

    # rest pose maybe need some edition
    # next step is to just interpolate the skinning weights for them

    # interpolated the weigths for the rest of the points
    completeMesh = pv.PolyData(outObjMeshFile)

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(outObjMeshFile, V, F)
    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()

    newWeights = np.array(jsonSkelData['Weights'])
    newWeights[:numCoarseJoints, :numRealPts] = np.array(jsonSkelDataCoarse['Weights'])[:, :numRealPts]

    igl.cotmatrix(V, F, L)
    LNP = - e2p(L).todense()
    nDimData = completeMesh.points.shape[0]
    # get points that are not on mesh
    isolatedVIds = getIsolatedVerts(completeMesh)
    # clean the LNP to make if singular:
    for isolV in isolatedVIds:
        LNP[isolV, isolV] = 1
    # 1. for reals joints, the hard constraints are the real points
    for iJ in tqdm.tqdm(range(numCoarseJoints), desc='for reals joints'):
        # build up constraint
        constraintIds = np.array(list(range(numRealPts)))
        constraintIds = np.unique(np.concatenate([constraintIds, isolatedVIds]))

        newWeights[iJ, :] = interpolateData(nDimData, newWeights[iJ, constraintIds], constraintIds, LNP)
        # if iJ in [14, 15, 9]:
        #     newWeights[iJ, vIdsNeedCorrsToSmplsh] = smplshWeights[toSmplshJointsCorrs[iJ], toSmplshVertsCorrs[:, 1]]
    numPts = completeMesh.points.shape[0]
    for iV in range(numPts):
        newWeights[:, iV] = newWeights[:, iV] / (np.sum(newWeights[:, iV]) if np.sum(newWeights[:, iV]) != 0 else 1)

    jsonSkelData['Weights'] = newWeights.tolist()
    # set up joints
    originalRestpose = completeMesh.points.tolist()
    for i in range(len(originalRestpose)):
        jsonSkelData['VTemplate'][0][i] = originalRestpose[i][0]
        jsonSkelData['VTemplate'][1][i] = originalRestpose[i][1]
        jsonSkelData['VTemplate'][2][i] = originalRestpose[i][2]
    originalJoints = pv.PolyData(inCoarseJoints)
    joints = originalJoints.points.tolist()
    for i in range(16):
        jsonSkelData['JointPos'][0][i] = joints[i][0]
        jsonSkelData['JointPos'][1][i] = joints[i][1]
        jsonSkelData['JointPos'][2][i] = joints[i][2]

    json.dump(jsonSkelData, open(outputNewSkelFile, 'w'))
    VisualizeVertRestPose(outputNewSkelFile, outputNewSkelFile+'.vtk', meshWithFaces=None, visualizeBoneActivation=False)
    VisualizeBones(outputNewSkelFile, outputNewSkelFile+'.Bone.vtk')
