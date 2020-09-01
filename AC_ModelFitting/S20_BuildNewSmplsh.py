import pyvista as pv
from scipy.spatial.transform import Rotation as R
SMPLSH_Dir = r'..\SMPL_reimp'
import sys
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch
import smpl_np
# import trimesh
from scipy.spatial import KDTree
import numpy as np
from iglhelpers import *
import pyigl as igl
import tqdm, os, json
from os.path import join
import vtk
from pathlib import Path
from SkelFit import Visualization
import pickle
from S19_SMPLHJToOPJointRegressor import *

def loadCompressedFittingParam(file, readPersonalShape=False):
    fitParam = np.load(file)
    transInit = fitParam['trans']
    poseInit = fitParam['pose']
    betaInit = fitParam['beta']

    if readPersonalShape:
        personalShape = fitParam['personalShape']
        return transInit, poseInit, betaInit, personalShape
    else:
        return transInit, poseInit, betaInit

def buildSMPLSocks(restPoseMeshF, skinningWeightsF, shapeBlendShapeF, poseBlendShapeF, JRegressorF,
        smplModelPath = r'C:/Data/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'):
    restPoseMesh = pv.PolyData(restPoseMeshF)

    restPoseV = np.copy(restPoseMesh.points)
    skinningWeights = np.load(skinningWeightsF)
    shapeBlendShape = np.load(shapeBlendShapeF)
    poseBlendShape = np.load(poseBlendShapeF)
    JRegressor = np.load(JRegressorF)

    faces = []
    fId = 0
    while fId < restPoseMesh.faces.shape[0]:
        numFVs = restPoseMesh.faces[fId]
        face = []
        fId += 1
        for i in range(numFVs):
            face.append(restPoseMesh.faces[fId])
            fId += 1

        faces.append(face)

    smpl = smpl_np.SMPLModel(smplModelPath)
    smpl.v_template = restPoseV
    smpl.weights = skinningWeights
    smpl.posedirs = poseBlendShape
    smpl.shapedirs = shapeBlendShape
    smpl.J_regressor = JRegressor
    smpl.faces = faces
    return smpl

def visualizeSMPLBones(smpl, outSkelVTK):
    numJoints = smpl.J.shape[0]

    ptsVtk = vtk.vtkPoints()
    # pts.InsertNextPoint(p1)
    for i in range(numJoints):
        ptsVtk.InsertNextPoint([smpl.J[i, 0], smpl.J[i, 1], smpl.J[i, 2]])

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(1, numJoints):
        # iParent = jData['Parents'].get(str(i))
        iParent = smpl.parent[i]
        if iParent != None:
            line = vtk.vtkLine()

            line.GetPointIds().SetId(0, i)  # the second 0 is the index of the Origin in the vtkPoints
            line.GetPointIds().SetId(1, iParent)  # the second 1 is the index of P0 in the vtkPoints
            lines.InsertNextCell(line)

    polyData.SetLines(lines)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outSkelVTK)
    writer.Update()

def searchForClosestPoints(sourceVs, tree):
    closestPts = []
    dis = []
    for sv in sourceVs:
        minDst = 100000
        closestP = None
        # for tv in targetVs:
        #     dst = np.linalg.norm(sv - tv)
        #     if dst < minDst:
        #         minDst = dst
        #         closestP = tv

        # dists = np.sum(np.square(targetVs - sv), axis=1)
        # tvId = np.argmin(dists)

        d, tvId = tree.query(sv)
        closestPts.append(tvId)
        dis.append(d)
    return np.array(closestPts), np.array(dis)

def buildKKT(L, D, e):
    nDimX = L.shape[0]
    nConstraints = D.shape[0]

    KKTMat = np.zeros((nDimX + nConstraints, nDimX + nConstraints))
    KKTMat[0:nDimX, 0:nDimX] = L
    KKTMat[nDimX:nConstraints + nDimX, 0:nDimX] = D
    KKTMat[0:nDimX, nDimX:nConstraints + nDimX] = np.transpose(D)

    KKTRes = np.zeros((nDimX + nConstraints,1))
    KKTRes[nDimX:nDimX + nConstraints,0] = e[:,0]

    return KKTMat, KKTRes

def interpolateData(nDimData, constrantData, constraintIds, LMat):
    # nDimData = constrantData.shape[0]
    nConstraints = constraintIds.shape[0]

    x = constrantData
    # Build Constraint
    D = np.zeros((nConstraints, nDimData))
    e = np.zeros((nConstraints, 1))
    for i, vId in enumerate(constraintIds):
        D[i, vId] = 1
        e[i, 0] = x[i]

    kMat, KRes = buildKKT(LMat, D, e)

    xInterpo = np.linalg.solve(kMat, KRes)

    # print("Spatial Laplacian Energy:",  xInterpo[0:nDimX, 0].transpose() @ LNP @  xInterpo[0:nDimX, 0])
    # wI = xInterpo[0:nDimX, 0]
    # wI[nConstraints:] = 1
    # print("Spatial Laplacian Energy with noise:",  wI @ LNP @  wI)

    return xInterpo[0:nDimData, 0]

def visualizeInterpolation(mesh, weights, vtkVisFile):
    weights = weights

    numJoints = weights.shape[1]

    for i in range(numJoints):
        mesh.point_arrays['Weight_%02i' % i] = weights[:, i]
        mesh.point_arrays['Weights_Log_%02i' % i] = np.log(np.abs(10e-16 + weights[:, i]))

    mesh.save(vtkVisFile)

def saveObj(path, verts, faces, faceIdAdd1 = True):
    with open(path, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in faces:
            if faceIdAdd1:
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
            else:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

class VertexToOpJointsConverter(torch.nn.Module):
    def __init__(s, **kwargs):
        super(VertexToOpJointsConverter, s).__init__(**kwargs)

        s.jSelector = VertexJointSelector(vertex_ids['smplsh'])
        jointMap = smpl_to_openpose('smplh', use_hands=True,
                                    use_face=False,
                                    use_face_contour=False, )

        s.joint_mapper = JointMapper(jointMap)

    def forward(s, smplshJoints):
        allJoints = s.jSelector(to_tensor(smplSocks.points[None, ...]), to_tensor(joints[None, ...]))
        joint_mapped = s.joint_mapper(allJoints)

        return joint_mapped


if __name__ == '__main__':
    # smpl_model_path = r'C:/Data/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

    # smpl_model_path = r'..\Data\BuildSmplsh\Input\SMPLH_male.pkl'
    # with open(smpl_model_path, 'rb') as smplh_file:
    #     model_data = pickle.load(smplh_file, encoding='latin1')
    #
    # inSMPLSocksMesh = r'..\Data\BuildSmplsh\Input\SMPLWithSocks_tri_Aligned.obj'
    # smplOriginAverageMeshOut = r'..\Data\BuildSmplsh\Output\SMPLOriginAverage.obj'
    #
    # interpoDataOutFolder = r'..\Data\BuildSmplsh\Output'
    # outBlendShapeInterpoFolder = r'..\Data\BuildSmplsh\Output\BlendShapes'
    # outSMPLSHNpzFile = join(interpoDataOutFolder, 'SmplshModel_m.npz')
    #
    # SMPLSocksTranslatedOut = r'SMPLWithSocks_tri_translated.ply'
    # outWeightVisFile = join(interpoDataOutFolder, 'WeightVisualization.vtk')

    # # Data for building smplsh male
    smpl_model_path = r'C:\Code\MyRepo\03_capture\Smpl_SeriesData\models\smplh\SMPLH_female.pkl'
    with open(smpl_model_path, 'rb') as smplh_file:
        model_data = pickle.load(smplh_file, encoding='latin1')

    # this should be aligned to smplh restpose
    inSMPLSocksMesh = r'..\Data\BuildSmplsh_Female\\InterpolateFemaleShape\SMPLWithSocks_tri_Aligned_female.obj'
    smplOriginAverageMeshOut = r'..\Data\BuildSmplsh_Female\Output\SMPLOriginAverage.obj'

    interpoDataOutFolder = r'..\Data\BuildSmplsh_Female\Output'
    outBlendShapeInterpoFolder = r'..\Data\BuildSmplsh_Female\Output\BlendShapes'

    # Data for building smplsh male
    outSMPLSHNpzFile = join(interpoDataOutFolder, 'SmplshModel_f.npz')

    SMPLSocksTranslatedOut = r'SMPLWithSocks_tri_translated.ply'
    outWeightVisFile = join(interpoDataOutFolder, 'WeightVisualization.vtk')


    os.makedirs(interpoDataOutFolder, exist_ok=True)
    os.makedirs(outBlendShapeInterpoFolder, exist_ok=True)

    # neither pyvista or trimesh supports the obj like one-vertex-multi-entries style data
    # they all have to duplicate some points
    # to make it work I have to only output the vertex position
    smplSocks = pv.PolyData(inSMPLSocksMesh)
    # smplSocks = trimesh.load(inSMPLSocksMesh)
    smplSocksRestPoseVFile = join(interpoDataOutFolder, 'SmplSocksRestPoseV.npy')
    np.save(smplSocksRestPoseVFile, smplSocks.points)

    # [SMPL vertex id, SMPL_Socks vertex id]
    correspondeces = [
        [331, 331]
    ]

    # First we need to find the correspondences between SMPL and modified SMPL_Socks
    # Somehow there is a translation between the SMPL np model and the SMPL maya model
    smplhOrgVertices = model_data['v_template']
    saveObj(smplOriginAverageMeshOut, smplhOrgVertices, model_data['f'])
    translation = smplhOrgVertices[correspondeces[0][0], :] - smplSocks.points[correspondeces[0][1], :]

    smplSocks.points += translation

    smplSocks.save(SMPLSocksTranslatedOut)

    # the vertex position are not exactly the same
    # we need to know the correspondence from smplSocks to smpl

    corrThreshold = 0.002
    tree = KDTree(smplhOrgVertices)
    corrs, ds = searchForClosestPoints(smplSocks.points, tree)

    constraintIds = np.where(ds <= corrThreshold)[0]
    goodCorrs = corrs[constraintIds]

    smplSToSmplCorrs = np.hstack([constraintIds.reshape(-1,1), goodCorrs.reshape(-1,1)])
    np.save(join(interpoDataOutFolder, 'SmplSToSmplCorrs.npy'), smplSToSmplCorrs)

    print(goodCorrs.shape)

    # We should build the interpolation matrix
    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(inSMPLSocksMesh, V, F)
    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()

    igl.cotmatrix(V, F, L)

    LNP = - e2p(L).todense()

    LNP = LNP @ LNP

    numJoints = model_data['J_regressor'].shape[0]
    nConstraints = goodCorrs.shape[0]
    nDimData = smplSocks.points.shape[0]

    newWeights = np.zeros((nDimData, model_data['J_regressor'].shape[0]))

    outSkinningWeightsFile = join(interpoDataOutFolder, 'SkinningWeightsInterpo.npy')

    # # interpolate skining weights
    # for iW in tqdm.tqdm(range(numJoints), desc= 'Interpolating Skinning Weights'):
    #     newWeights[:, iW] = interpolateData(nDimData, model_data['weights'][goodCorrs, iW], constraintIds, LNP)
    # np.save(outSkinningWeightsFile, newWeights)
    # visualizeInterpolation(smplSocks, newWeights, outWeightVisFile)
    #
    outShapeBlendShapesFile = join(interpoDataOutFolder, 'ShapeBlendShapesInterpo.npy')
    # # interpolate shape blend shapes
    # numShapeParameter = model_data['shapedirs'].shape[2]
    # # Shape blend shapes: nVerts x 3 x 10
    # newSBS = np.zeros((nDimData, 3, model_data['shapedirs'].shape[2]))
    # shapeBlendShapeOutFolder = join(outBlendShapeInterpoFolder, "ShapeBlendShapes")
    # os.makedirs(shapeBlendShapeOutFolder, exist_ok=True)
    # for iS in tqdm.tqdm(range(numShapeParameter), desc= 'Interpolating Pose Blend Shapes'):
    #     for iDim in range(3):
    #         newSBS[:, iDim, iS] = interpolateData(nDimData, model_data['shapedirs'][goodCorrs, iDim, iS], constraintIds, LNP)
    #     verts = smplSocks.points + 3*newSBS[:, :, iS]
    #     sbsMesh = pv.PolyData(inSMPLSocksMesh)
    #     sbsMesh.points = verts
    #     outBlendShapeFile = join(shapeBlendShapeOutFolder, 'SBS' + str(iS).zfill(3) + '.ply')
    #     sbsMesh.save(outBlendShapeFile)
    #
    # np.save(outShapeBlendShapesFile, newSBS)
    #
    outPoseBlendShapesFile = join(interpoDataOutFolder, 'SMPLSH_PoseBlendShapes.npy')
    # # interpolate pose blend shapes
    # numPoseParameter = model_data['posedirs'].shape[2]
    # # # Shape blend shapes: nVerts x 3 x 207
    # newPBS = np.zeros((nDimData, 3, numPoseParameter))
    # poseBlendShapeOutFolder = join(outBlendShapeInterpoFolder, "PoseBlendShapes")
    # os.makedirs(poseBlendShapeOutFolder, exist_ok=True)
    # for iP in tqdm.tqdm(range(numPoseParameter), desc= 'Interpolating Pose Blend Shapes'):
    #     for iDim in range(3):
    #         newPBS[:, iDim, iP] = interpolateData(nDimData, model_data['posedirs'][goodCorrs, iDim, iP], constraintIds, LNP)
    #
    #     verts = smplSocks.points + 3*newPBS[:, :, iP]
    #     sbsMesh = pv.PolyData(inSMPLSocksMesh)
    #     sbsMesh.points = verts
    #     outBlendShapeFile = join(poseBlendShapeOutFolder, 'PBS' + str(iP).zfill(3) + '.ply')
    #     sbsMesh.save(outBlendShapeFile)
    #
    # np.save(outPoseBlendShapesFile, newPBS)
    # #
    outJRegressorFile = join(interpoDataOutFolder, 'SMPLSH_J_regressor.npy')
    # # There is one last J regressor to interpolate
    # # For J regressor we really cannot use interpolation
    # # numJRegressor = smpl.J_regressor.shape[0]
    # # # # Shape blend shapes: nJoint x nVerts
    # # newJRegressor = np.zeros((numJRegressor, nDimData))
    # #
    # # for iJR in tqdm.tqdm(range(numJRegressor), desc='Interpolating Joint Regressor'):
    # #     data = np.copy(((smpl.J_regressor[iJR, goodCorrs]).todense()))
    # #     data = np.squeeze(data)
    # #     newJRegressor[iJR, :] = interpolateData(nDimData, data, constraintIds, LNP)
    # # np.save(outJRegressorFile, newJRegressor)
    #
    # # Joint regressor is basically a weighted average of SMPL vertices
    # # Since we have less vertices on the feet, we will need to rebalance the weights
    # # My solution is, for every point on smpl, add its weight to corresponding closest point on SMPLSocks
    treeSMPLS = KDTree(smplSocks.points)
    corrsSMPLToSMPLSocks, ds = searchForClosestPoints(smplhOrgVertices, treeSMPLS)
    numJRegressor = model_data['J_regressor'].shape[0]
    # # Shape blend shapes: nVerts x 3 x 207
    newJRegressor = np.zeros((numJRegressor, nDimData))
    JRegressorData = np.copy(model_data['J_regressor'].todense())
    for iVSMPL, iVSMPLSocks in enumerate(corrsSMPLToSMPLSocks):
        newJRegressor[:, iVSMPLSocks] += JRegressorData[:, iVSMPL]
    np.save(outJRegressorFile, newJRegressor)

    joints = newJRegressor @ smplSocks.points
    jointsPC = pv.PolyData(joints)
    jointsPC.save(join(interpoDataOutFolder, 'SmplshJoints.ply'))

    # visualizeSMPLBones(smpl, 'SMPLBones.vtk')
    # The joint selector for smplsh

    smplhVertexIds = vertex_ids['smplh']
    newIdMap = {}
    for key, value in smplhVertexIds.items():
        # print(key, ':', value, '->', corrsSMPLToSMPLSocks[value])
        newIdMap[key] = int(corrsSMPLToSMPLSocks[value])

    # print(newIdMap)
    smplshJointSelectionMapFile = join(interpoDataOutFolder, 'SMPLSH_JointSelector.json')
    json.dump(newIdMap, open(smplshJointSelectionMapFile, 'w'), indent=3)

    # Test the joint selector for new smplsh

    newJRegressor = np.load(outJRegressorFile)
    joints = newJRegressor @ smplSocks.points
    jSelector = VertexJointSelector(vertex_ids['smplsh'])
    allJoints = jSelector(to_tensor(smplSocks.points[None,...]), to_tensor(joints[None,...]))

    jointMap = smpl_to_openpose('smplh', use_hands=True,
                                use_face=False,
                                use_face_contour=False, )

    print(jointMap)
    joint_mapper = JointMapper(jointMap)
    joint_mapped = joint_mapper(allJoints)
    allJointsPC = pv.PolyData(joint_mapped.detach().cpu().numpy())
    allJointsPC.save(join(interpoDataOutFolder, 'AllJoints.ply'))

    np.savez_compressed(outSMPLSHNpzFile, VTemplate=np.array(smplSocks.points).astype(np.float64), Faces=smplSocks.faces, Weights=np.load(outSkinningWeightsFile).astype(np.float64),
             ShapeBlendShapes=np.load(outShapeBlendShapesFile).astype(np.float64),
             PoseBlendShapes=np.load(outPoseBlendShapesFile).astype(np.float64), JRegressor=newJRegressor.astype(np.float64), ParentTable=np.load(join(interpoDataOutFolder, 'SMPLSH_ParentsTable.npy')))

    # test new smplsh data
    inFittingParam = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03052.npz'

    device = torch.device("cpu")
    # torch.cuda.set_device(device)

    transInit, poseInit, betaInit = loadCompressedFittingParam(inFittingParam, readPersonalShape=False)
    # Make fitting parameter tensors
    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=False, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=False, device=device)
    smplsh = smplsh_torch.SMPLModel(device, outSMPLSHNpzFile, personalShape=None)
    verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)

    newJRegressor = torch.tensor(newJRegressor, dtype=torch.float64, requires_grad=False, device=device)
    jointsRegressed = newJRegressor @ verts

    jointDiff = jointsDeformed - jointsRegressed

    saveObj(join(interpoDataOutFolder, 'TestSMPLSH.obj'), verts, smplsh.faces)





