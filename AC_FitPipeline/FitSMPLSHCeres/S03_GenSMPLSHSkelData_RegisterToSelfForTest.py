import json, sys
from Utility import *
SMPLSH_Dir = r'..\SMPL_reimp'
sys.path.insert(0, SMPLSH_Dir)
import smplsh_np
import numpy as np
import pyvista as pv

def padOnes(mat):
    return np.vstack([mat, np.ones((1, mat.shape[1]))])
if __name__ == '__main__':
    # What is needed: SMPLSH data
    # Registration Matrix

    SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    toSparsePCMat = r'..\Data\PersonalModel_Lada\InterpolationMatrix.npy'
    registrationTIdFile = r'..\Data\PersonalModel_Lada\InterpolationTriId.npy'
    registrationBarysFile = r'..\Data\PersonalModel_Lada\InterpolationBarys.npy'
    toSparsePCMat = r'..\Data\PersonalModel_Lada\InterpolationMatrix.npy'
    betaFile = r'..\Data\PersonalModel_Lada\BetaFile.npy'
    personalShapeFile = r'..\Data\PersonalModel_Lada\PersonalShape.npy'
    smplshMeshFile = r'..\Data\PersonalModel_Lada\FinalMesh.obj'
    smplshMeshQuadFile = r'..\Data\BuildSmplsh\SMPLWithSocks_Quad_xyzOnly.obj'
    # the initial Skel data
    # outSkelDataName = r'..\Data\PersonalModel_Lada\SmplshSkelData\01_SmplshSkelData_Lada.json'
    outSkelDataName = r'..\Data\PersonalModel_Lada\SmplshSkelData\02_SmplshSkelDataRegisterToSelf_Lada.json'

    # data = np.load(model_path)
    # self.J_regressor = data['JRegressor']
    # self.weights = data['Weights']
    # self.posedirs = data['PoseBlendShapes']
    # self.v_template = data['VTemplate']
    # self.shapedirs = data['ShapeBlendShapes']
    # self.faces = data['Faces']
    # self.parent = data['ParentTable']

    smplsh =  smplsh_np.SMPLSHModel(SMPLSHNpzFile)
    betas = np.load(betaFile)
    registrationMatrix = np.load(toSparsePCMat)
    # tIds = np.load(registrationTIdFile)
    # barys = np.load(registrationBarysFile)
    # jointLocations = smplsh_np.

    smplsh.set_params(beta = betas)
    smplsh.update()

    # get the smplsh data
    jointLocations = smplsh.J
    vTemplate = smplsh.v_template
    parents = smplsh.parent
    # kintree_table = smplsh.
    parentsMap = {i:int(parents[i]) for i in range(len(parents))}

    smplshMeshQuad = pv.PolyData(smplshMeshQuadFile)
    facesQuad = []
    fId = 0
    while fId < smplshMeshQuad.faces.shape[0]:
      numFVs = smplshMeshQuad.faces[fId]
      face = []
      fId += 1
      for i in range(numFVs):
        face.append(int(smplshMeshQuad.faces[fId]))
        fId += 1

      facesQuad.append(face)

    numJoints = jointLocations.shape[0]
    registrationVids = []
    barys = []

    for i in range(registrationMatrix.shape[0]):
        registrationVids.append(np.where(registrationMatrix[i, :]))
        barys.append(registrationMatrix[i, np.where(registrationMatrix[i, :])])

    skelData = {
        'VTemplate': padOnes(1000 * vTemplate.transpose()).tolist(),
        'Weights': smplsh.weights.transpose().tolist(),
        'JointPos': padOnes(1000 * jointLocations.transpose()).tolist(),
        'Faces': facesQuad,
        'KintreeTable': [],
        'Parents': parentsMap,
        "TriVidsNp": [[i,i,i] for i in range(vTemplate.shape[0])],
        "BarycentricsNp": [[0.33333333,0.33333333,0.33333334] for i in range(vTemplate.shape[0])],
        "ActiveBoneTable": [list(range(numJoints)) for i in range(vTemplate.shape[0])],
        "HeadJointIds": [12, 15],
        "HandJointIds": list(range(22,52))
    }
    json.dump(skelData, open(outSkelDataName, 'w'), indent=2)





