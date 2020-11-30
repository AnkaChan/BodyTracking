from Utility import *
import numpy as np

if __name__ == '__main__':
    # smplshMesh_Lata = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\TextureMap2Color\SMPLWithSocks_tri.obj'
    # outLapMat = r'SmplshRestposeLapMat_Lada.npy'

    smplshMesh_Lata = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\KateyBodyModel\BodyMesh\Initial\Katey.obj'
    outLapMat = r'SmplshRestposeLapMat_Katey.npy'


    LNP = getLaplacian(smplshMesh_Lata)

    print("Condition number before washing: ", np.linalg.cond(LNP))

    LNP[np.where(np.logical_and(LNP < 0.01, LNP >0) )] = 0.01
    LNP[np.where(np.logical_and(LNP > -0.01, LNP < 0) )] = -0.01
    LNP = LNP + 0.01 * np.eye(LNP.shape[0])

    print("Condition number before washing: ", np.linalg.cond(LNP))

    np.save(outLapMat, LNP)
