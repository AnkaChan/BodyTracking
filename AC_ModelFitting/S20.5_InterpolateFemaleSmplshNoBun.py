SMPLSH_Dir = r'..\SMPL_reimp'
import sys
sys.path.insert(0, SMPLSH_Dir)
import smplsh_torch
import smpl_np
import numpy as np
from iglhelpers import *
import pyigl as igl
import pickle
import pyvista as pv
from scipy.spatial import KDTree
from S20_BuildNewSmplsh import *

if __name__ == '__main__':
    # Need aligned male smplsh mesh, smplh male restpose mesh, smplh female restpose mesh
    alignedMaleSmplshMeshFile = '..\Data\BuildSmplsh_Female\InterpolateFemaleShape\SMPLWithSocks_tri_Aligned_male.obj'
    smpl_model_male_path = r'C:\Code\MyRepo\03_capture\Smpl_SeriesData\models\smplh\SMPLH_male.pkl'
    smpl_model_female_path = r'C:\Code\MyRepo\03_capture\Smpl_SeriesData\models\smplh\SMPLH_female.pkl'

    outAlignedFemaleSmplshMeshFile = '..\Data\BuildSmplsh_Female\InterpolateFemaleShape\SMPLWithSocks_tri_Aligned_female_NoBun.ply'
    # bunIdFile = r'BunIndices.json'
    bunIdFile = r'BunIndices_Smaller.json'
    bunIds = json.load(open(bunIdFile))


    with open(smpl_model_male_path, 'rb') as smplh_file:
        model_data_male = pickle.load(smplh_file, encoding='latin1')

    with open(smpl_model_female_path, 'rb') as smplh_file:
        model_data_female = pickle.load(smplh_file, encoding='latin1')

    maleRestposeVerts = model_data_male['v_template']
    femaleRestposeVerts = model_data_female['v_template']
    smplhFace = model_data_female['f']

    saveObj(r'..\Data\BuildSmplsh_Female\InterpolateFemaleShape\smplhFemaleRestpose.obj', femaleRestposeVerts, smplhFace)

    smplshMale = pv.PolyData(alignedMaleSmplshMeshFile)

    # print(maleRestposeVerts - smplshMale.points)

    corrThreshold = 0.002
    tree = KDTree(maleRestposeVerts)
    corrsSmplshToSmpl, ds = searchForClosestPoints(smplshMale.points, tree)

    constraintIds = np.where(ds <= corrThreshold)[0]
    constraintIds = np.setdiff1d(constraintIds, bunIds)

    corrsSmplshToSmpl = corrsSmplshToSmpl[constraintIds]

    print(maleRestposeVerts[corrsSmplshToSmpl, :] - smplshMale.points[constraintIds, :])

    # We should build the interpolation matrix
    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(alignedMaleSmplshMeshFile, V, F)
    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()

    igl.cotmatrix(V, F, L)

    LNP = - e2p(L).todense()
    LNP = LNP @ LNP

    displacementConstraints = femaleRestposeVerts[corrsSmplshToSmpl, :] - smplshMale.points[constraintIds, :]

    femaleSmplshVerts = np.zeros(smplshMale.points.shape)
    for i in range(3):
        coordInterpolated = interpolateData(smplshMale.points.shape[0], displacementConstraints[:, i], constraintIds, LNP)
        femaleSmplshVerts[:, i] = coordInterpolated

    smplshMale.points = smplshMale.points+femaleSmplshVerts
    smplshMale.save(outAlignedFemaleSmplshMeshFile)

