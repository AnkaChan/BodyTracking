from SkelFit.Data import *
from Utility import *
from scipy.spatial.transform import Rotation as R
import sys, os, tqdm
SMPLSH_Dir = r'..\SMPL_reimp'
sys.path.insert(0, SMPLSH_Dir)
import smplsh_np
import pyvista as pv
from pathlib import Path
import subprocess

if __name__ == '__main__':
    # # Not change for Lada
    # personalShapeFile = r'..\Data\PersonalModel_Lada\PersonalShape.npy'
    # betaFile = r'..\Data\PersonalModel_Lada\BetaFile.npy'
    # SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    # smplshMeshQuadFile = r'..\Data\BuildSmplsh\SMPLWithSocks_Quad_xyzOnly.obj'
    #
    # # # Need to be changed every sequence
    # # inputCoarseMeshChunkFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Lada_Ground.json'
    # # smplshSkelDataFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\PersonalModel_Lada\SmplshSkelData\01_SmplshSkelData_Lada.json'
    # # inputParamsFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Fit\outputs\Params'
    # # outFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Fit\WithPBS'
    #
    # # Need to be changed every sequence
    # inputCoarseMeshChunkFile = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Inputs\Lada_Stand.json'
    # smplshSkelDataFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\PersonalModel_Lada\SmplshSkelData\01_SmplshSkelData_Lada.json'
    # smplshFitOutputFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Fit\Lada_Stand'
    # # interpolatedOutFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Interpolated\Fit\WithPBS'

    # Not change for Katey
    personalShapeFile =  r'..\Data\KateyBodyModel\PersonalShape.npy'
    betaFile = r'..\Data\KateyBodyModel\beta.npy'
    SMPLSHNpzFile =  r'..\Data\BuildSmplsh_Female\Output\SmplshModel_f_noBun.npz'
    smplshMeshQuadFile = r'..\Data\BuildSmplsh\SMPLWithSocks_Quad_xyzOnly.obj'

    # Need to be changed every sequence
    inputCoarseMeshChunkFile = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Inputs\Katey_Stand.json'
    smplshSkelDataFile =  r'..\Data\KateyBodyModel\SmplshSkelData\01_SmplshSkelData_Katey.json'
    smplshFitOutputFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Fit\Katey_Stand'
    # smplshFitOutputFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Fit\Katey_Stand2'

    deformedWithPBSFolder = join(smplshFitOutputFolder, 'WithPBS')
    os.makedirs(smplshFitOutputFolder, exist_ok=True)
    os.makedirs(deformedWithPBSFolder, exist_ok=True)

    # # run C++ smplsh fitting code
    # subprocess.call(
    #     ['Bin\\SMPLSHFitToPointCloudDynamicAccelerated', inputCoarseMeshChunkFile, smplshFitOutputFolder, '-s', smplshSkelDataFile, '-c'])

    inputParamsFolder = join(smplshFitOutputFolder, r'Params')

    personalShape = np.load(personalShapeFile) / 1000
    beta = np.load(betaFile)

    paramFiles = sortedGlob(join(inputParamsFolder, '*.json'))

    smplshModel = smplsh_np.SMPLSHModel(SMPLSHNpzFile)
    exampleMesh = pv.PolyData(smplshMeshQuadFile)

    for paramF in tqdm.tqdm(paramFiles):
        quaternions, translation = loadPoseFile(paramF)
        fileName = Path(paramF).stem

        Rs = [R.from_quat([q[1], q[2], q[3], q[0]]) for q in quaternions]
        Rs = np.array([r.as_rotvec() for r in Rs])

        verts = smplshModel.set_params(pose=Rs, beta=beta, trans=np.array(translation) / 1000, personalShape=personalShape)
        # pose = np.zeros((24, 3))
        #
        # pose[preservedJoints, :] = Rs
        exampleMesh.points = verts * 1000

        outFile = join(deformedWithPBSFolder, fileName + '.ply')
        exampleMesh.save(outFile)




