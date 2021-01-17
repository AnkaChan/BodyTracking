import numpy

import sys, os
from os.path import  join
import json
import math
import numpy as np

import pyigl as igl
import pyvista as pv
from pathlib import Path

import random
from iglhelpers import *
from SkelFit.Visualization import *
from SkelFit.Data import *


def padOnes(mat):
    return np.vstack([mat, np.ones((1, mat.shape[1]))])

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


if __name__ == '__main__':

    inMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2_no_hole_tri.obj'
    outClearedMesh = r'F:\WorkingCopy2\2020_12_23_Marianne_BodyModel\FinalMesh_Marianne_OnlyQuad\Mesh_OnlyQuad_edited2_no_hole_tri_clear.obj'
    skelFile = r'F:\WorkingCopy2\2019_12_24_Marianne_Capture\ActorCalibration\2\Fit029\Model\Skel0059.json'
    outSkelData = r'01_SkelData_Marianne_Complete_newMesh.json'

    # clear vertex that are not on any faces
    meshWithFaces = pv.PolyData(inMesh)
    vertsOnFaces = set()
    faces = []
    fId = 0
    while fId < meshWithFaces.faces.shape[0]:
        numFVs = meshWithFaces.faces[fId]
        face = []
        fId += 1
        for i in range(numFVs):
            face.append(meshWithFaces.faces[fId])
            vertsOnFaces.add(meshWithFaces.faces[fId])
            fId += 1

        faces.append(face)

    for i in range(meshWithFaces.points.shape[0]):
        if i not in vertsOnFaces:
            meshWithFaces.points[i, :] = [0,0,-1]

    write_obj(outClearedMesh, meshWithFaces.points, faces)

    meshData = pv.PolyData(outClearedMesh)
    skelData = json.load(open(skelFile))

    V = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    igl.readOBJ(outClearedMesh, V, F)

    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()
    igl.cotmatrix(V, F, L)

    LNP = - e2p(L).todense()

    print("Max value in Spatial Laplacian:", np.max(np.abs(LNP)))

    assert np.max(np.abs(LNP)) < 10

    weights = np.array(skelData['Weights'])
    numJoints = weights.shape[0]
    nDimX = meshData.points.shape[0]
    # constraints are all the old points (for isolated vertex on rest pose we need to set it fixed, otherwise KKT becomes singular)
    nConstraints = weights.shape[1]

    newWeights = np.zeros((numJoints, nDimX))
    # print(LNP[np.where(meshData.points[:, 2]==-1)[0], :])
    LNP[np.where(meshData.points[:, 2] == -1), np.where(meshData.points[:, 2] == -1)] = 1
    for iW in range(numJoints):
        x = weights[iW, :]
        # Build Constraint
        D = np.zeros((nConstraints, nDimX))
        e = np.zeros((nConstraints, 1))
        for vId in range(nConstraints):
            D[vId, vId] = 1
            e[vId, 0] = x[vId]

        kMat, KRes = buildKKT(LNP, D, e)

        xInterpo = np.linalg.solve(kMat, KRes)

        # print("Spatial Laplacian Energy:",  xInterpo[0:nDimX, 0].transpose() @ LNP @  xInterpo[0:nDimX, 0])
        # wI = xInterpo[0:nDimX, 0]
        # wI[nConstraints:] = 1
        # print("Spatial Laplacian Energy with noise:",  wI @ LNP @  wI)

        newWeights[iW, :] = xInterpo[0:nDimX, 0]

    skelData['Weights'] = newWeights.tolist()
    skelData['VTemplate'] = padOnes(meshData.points.transpose()).tolist()

    parent = skelData['Parents']
    parent = {int(key[0]): key[1] for key in parent.items()}

    jointsIdToReduce = [10, 11, 12, 15, 20, 21, 22, 23]

    numActiveJoints = weights.shape[1] - len(jointsIdToReduce)
    allJoints = list(range(0, 24))
    preservedJoints = allJoints
    for id in jointsIdToReduce:
        preservedJoints.remove(id)

    oldJIdToNewJId = {oldJId: i for i, oldJId in enumerate(preservedJoints)}
    new_kintree_table = [[0, -1]]

    for pair in skelData['KintreeTable']:
        if pair[0] != -1 and pair[1] != -1:
            new_kintree_table.append([oldJIdToNewJId.get(pair[0], -1), oldJIdToNewJId.get(pair[1], -1)])

    newFs = e2p(F)
    skelData['Faces'] = newFs.tolist()

    skelData['ActiveBoneTable'] = [list(range(16)) for i in range(nDimX)]
    skelData['KintreeTable'] = new_kintree_table
    skelData['Parents'] = parent

    json.dump(skelData, open(outSkelData, 'w'), indent=2)

    VisualizeVertRestPose(outSkelData, outSkelData + ".vtk",
                          visualizeBoneActivation=True,
                          addLogScaleWeights=True,
                          meshWithFaces=outClearedMesh)
    VisualizeBones(outSkelData, outSkelData + ".Bone.vtk")