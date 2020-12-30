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
from SkelFit.Geometry import *
from SkelFit.SkeletonModel import *

if __name__ == '__main__':
    inSmplshSkelData = r'..\Data\PersonalModel_Lada\SmplshSkelData\01_SmplshSkelData_Lada.json'
    inCoarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    inDeformedSmpsh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\03052_OnlyXYZ.obj'
    inCompletedMesh = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly.obj'
    headVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HeadVIds.Json'
    handVIdsFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\HandVIds.json'

    outFolder = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487'
    outCorseJointsFile = join(outFolder, 'CoarseJoints.obj')
    outSmplshJointsFile = join(outFolder, 'SMPLSHJoints.obj')

    outputNewSkelFile = join(outFolder, 'S01_Combined_Lada_HandHead.json')

    # F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0
    parameterFileCoarse = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\LBSWithTC\Params\A00003052.txt'
    parameterFileSmplsh = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\Param_00499.npz'

    smplshSkelData = json.load(open(inSmplshSkelData))
    coarseSkelData = json.load(open(inCoarseSkelData))

    minDis = 1
    numRealPts = 1487

    VisualizeVertRestPose(inSmplshSkelData, inSmplshSkelData+'.vtk', meshWithFaces=None)

    corrToRealPts = [i for i in range(numRealPts)]

    # find the corrspondences to smplsh mesh
    completeMesh = pv.PolyData(inCompletedMesh)
    smplshMesh = pv.PolyData(inDeformedSmpsh)

    headVIds = json.load(open(headVIdsFile))
    handVIds = json.load(open(handVIdsFile))

    vIdsNeedCorrsToSmplsh = headVIds + handVIds
    toSmplshVertsCorrs = []
    for VId in vIdsNeedCorrsToSmplsh:
        p = completeMesh.points[VId, :]
        diff = smplshMesh.points-p
        dis = np.sqrt(diff[:,0]**2+diff[:,1]**2+diff[:,2]**2)

        closestPt = np.argmin(dis)

        if dis[closestPt] > minDis:
            print('Warning: min distance: ', dis[closestPt])

        toSmplshVertsCorrs.append([VId, closestPt])

    # set up new joints and weighjts:
    smplshWeights = np.array(smplshSkelData['Weights']) # 52 rows, 6750 cols
    coarseWeights = np.array(coarseSkelData["Weights"])

    numCoarseJoints = len(coarseWeights)
    totalNumJoints = (numCoarseJoints  # coarse
                    + 2  # two head joints
                    + 2  # wrist: smplsh 20, 21
                    + 30 # hands
                    )
    numPts = completeMesh.points.shape[0]

    jointsIdToReduce = [10, 11, 12, 15, 20, 21, 22, 23] # joints we removed from
    toSmplshJointsCorrs = []
    iJCoarse = 0
    for iJ in range(24): # smpl has 24 joints
        if iJ not in jointsIdToReduce:
            toSmplshJointsCorrs.append([iJCoarse, iJ])
            iJCoarse += 1

    toSmplshJointsCorrs = toSmplshJointsCorrs + [[numCoarseJoints, 12], [numCoarseJoints+1, 15]] + [[numCoarseJoints + 2 + i, 20 + i] for i in range(32)]
    smplshToNewJointsCorrs = {jSmplsh[1]:jSmplsh[0] for jSmplsh in toSmplshJointsCorrs}

    newWeights = np.zeros((totalNumJoints, numPts), dtype=np.float64)

    # for first numRealPts verts, set their first 16 weights to identical as before
    newWeights[:numCoarseJoints, :numRealPts] = coarseWeights[:, :numRealPts]

    # for points corresponds to smplsh, copy the weights (including the coarse joints, because they also corresponds to smplsh joints):
    toSmplshJointsCorrs = np.array(toSmplshJointsCorrs)[:,1]
    toSmplshVertsCorrs = np.array(toSmplshVertsCorrs)
    newWeights[:, toSmplshVertsCorrs[:,0]] = smplshWeights[toSmplshJointsCorrs[:, None],  toSmplshVertsCorrs[:,1]]

    # print(toSmplshVertsCorrs)

    # get the joint position:
    # two steps, coarse model joints and smplsh Joints, I have deform the model using the parameters to obtain the new joints posisiton
    #   for coarse model

    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(inCoarseSkelData)
    r, t = readSkelParams(parameterFileCoarse)
    Rs = quaternionsToRotations(r)

    # newJs = transformJoints(Rs,t, J, parent)
    # print('newJs', newJs)
    #
    # # write_obj(outCorseJointsFile, newJs, None)
    newJsCoarse = pv.PolyData(outCorseJointsFile)

    parameterSmplsh = np.load(parameterFileSmplsh)
    vRestpose, J, weights, poseBlendShape, kintreeTable, parent, faces = readSkeletonData(inSmplshSkelData)
    # r, t = readSkelParams(parameterFileCoarse)
    Rs = axisAnglesToRotation(parameterSmplsh['pose'].reshape((-1, 3)))
    newJsSMPLSH = transformJoints(Rs, parameterSmplsh['trans'], J, parent)
    # print('newJs', newJs)
    write_obj(outSmplshJointsFile, newJsSMPLSH, None)

    # Joint kinematics
    parentTableCoarse = coarseSkelData['Parents']
    parentTableSmplsh = smplshSkelData['Parents']

    # head
    parentTableCoarse[str(numCoarseJoints + 1)] = int(numCoarseJoints)
    parentTableCoarse[str(numCoarseJoints)] = 9

    parentTableCoarse[str(numCoarseJoints+2)] = int(numCoarseJoints-2) # left hand
    parentTableCoarse[str(numCoarseJoints+3)] = int(numCoarseJoints-1) # right hand

    # hands
    for iJ in range(numCoarseJoints+4, totalNumJoints):
        smplshJoint = toSmplshJointsCorrs[iJ]
        smplshParent = parentTableSmplsh[str(smplshJoint)]
        newParent = smplshToNewJointsCorrs[smplshJoint]
        parentTableCoarse[str(iJ)] = int(newParent)

    print(parentTableCoarse)
    newSkelData = json.load(open(inCoarseSkelData))

    # obtain the joints
    newJoints = np.vstack([newJsCoarse.points, newJsSMPLSH[[12, 15], :], newJsSMPLSH[20:, :]])

    newSkelData['VTemplate'] =  padOnes( completeMesh.points.transpose()).tolist()
    newSkelData['JointPos'] =  padOnes( newJoints.transpose()).tolist()
    newSkelData['Weights'] = newWeights.tolist()
    newSkelData['Parents'] = parentTableCoarse
    newSkelData['Faces'] = retrieveFaceStructure(completeMesh)

    json.dump(newSkelData, open(outputNewSkelFile, 'w'))

    VisualizeVertRestPose(outputNewSkelFile, outputNewSkelFile+'.vtk', meshWithFaces=None)

