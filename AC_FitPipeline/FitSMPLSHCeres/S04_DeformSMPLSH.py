from SkelFit.Data import *
from Utility import *
from scipy.spatial.transform import Rotation as R
import sys, os, tqdm
SMPLSH_Dir = r'..\SMPL_reimp'
sys.path.insert(0, SMPLSH_Dir)
import smplsh_np
import pyvista as pv
from pathlib import Path

if __name__ == '__main__':
    inputParamsFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Fit\outputs\Params'
    personalShapeFile = r'..\Data\PersonalModel_Lada\PersonalShape.npy'
    betaFile = r'..\Data\PersonalModel_Lada\BetaFile.npy'
    SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
    outFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Fit\WithPBS'
    smplshMeshQuadFile = r'..\Data\BuildSmplsh\SMPLWithSocks_Quad_xyzOnly.obj'

    os.makedirs(outFolder, exist_ok=True)

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

        outFile = join(outFolder, fileName + '.ply')
        exampleMesh.save(outFile)


