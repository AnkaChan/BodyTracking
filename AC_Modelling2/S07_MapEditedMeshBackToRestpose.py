from SkelFit.Data import *

import pyvista as pv
from M01_ARAPDeformation import ARAPDeformation
if __name__ == '__main__':
    # inCoarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    # inParamData = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\LBSWithTC\params\A00003052.txt'
    #
    #
    # rs,ts = readSkelParams(inParamData)
    parameterFileCoarse = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\LBSWithTC\Params\A00003052.txt'
    parameterFileSmplsh = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\Param_00499.npz'

    inTargetObjMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json.obj'
    inSourceObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri.obj'
    outObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\Complete_withHeadHand_XYZOnly_tri_backToRestpose.obj'
    inCoarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'


    # try ARAP deformation
    # inSourceObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\CompleteBetterFeet_tri.obj'
    # outObjMeshFile = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487\CompleteBetterFeet_tri_backToRestpose.obj'

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

    ARAPDeformation(inSourceObjMeshFile, inTargetObjMeshFile, outObjMeshFile, corrs)

    # ARAP Works, but it only works for the mesh, and it performs badly on the head
    # So I still need to map every thing back to restpose

    # for hand and head I have to use inverseDeformation
    # rs, ts = readSkelParams(parameterFileCoarse)

