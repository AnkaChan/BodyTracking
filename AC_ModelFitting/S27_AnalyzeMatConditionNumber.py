import numpy as np
import pyvista as pv
from S23_2_Pipeline_IntialToSilhouetteFitting_Subdivision import *
from S05_InterpolateWithSparsePointCloud import *

if __name__ == '__main__':
    inRestposeSMPLSHMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\BuildSmplsh_Female\InterpolateFemaleShape\SMPLWithSocks_tri_Aligned_female_NoBun.obj'

    LNP = getLaplacian(inRestposeSMPLSHMesh, biLaplacian=False)

    print("Average none 0 value in LNP:", np.mean(np.abs(LNP[np.where(LNP)])))

    print('Condition number of LNP:', np.linalg.cond(LNP))

    print('Condition number of LNP blended:', np.linalg.cond(LNP + 0.01 * np.eye(LNP.shape[0])))


