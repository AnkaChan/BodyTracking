import tqdm
import glob, os
from os.path import join
import numpy as np
from pathlib import Path
def sortedGlob(pathname):
    return sorted(glob.glob(pathname))

def getLaplacian(meshFile, biLaplacian = False):
    import pyigl as igl
    import iglhelpers

    extName = Path(meshFile).suffix
    V = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()

    if extName.lower() == '.obj':
        igl.readOBJ(meshFile, V, F)
    elif extName.lower()  == '.ply':
        N = igl.eigen.MatrixXd()
        UV = igl.eigen.MatrixXd()
        igl.readPLY(meshFile, V, F, N, UV)

    # Compute Laplace-Beltrami operator: #V by #V
    L = igl.eigen.SparseMatrixd()

    igl.cotmatrix(V, F, L)

    LNP = - iglhelpers.e2p(L).todense()
    if biLaplacian:
        LNP = LNP @ LNP

    return LNP
