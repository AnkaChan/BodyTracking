from os.path import join
import glob
import os
from pathlib import Path
from iglhelpers import *

def sortedGlob(pathname):
    return sorted(glob.glob(pathname))

def getLaplacian(meshFile, biLaplacian = False):
    import pyigl as igl

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

    LNP = - e2p(L).todense()
    if biLaplacian:
        LNP = LNP @ LNP

    return LNP