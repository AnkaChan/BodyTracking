import M03_ToSparseFitting
from S11_ToSparseFittingSelectedFramesV2 import InputBundle
from Utility import *
from pathlib import Path
from M04_ObjConverter import *
import tqdm, os

if __name__ == '__main__':
    inputDeformedSmplshFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Fit\WithPBS'
    inputCoarseMeshFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Final_Smoothed1'
    outFolder = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Interpolated'

    os.makedirs(outFolder, exist_ok=True)

    inputs = InputBundle()
    inputs.laplacianMatFile = 'SmplshRestposeLapMat_Lada.npy'

    deformedMeshFiles = sortedGlob(join(inputDeformedSmplshFolder, '*.ply'))
    coarseMeshFiles = sortedGlob(join(inputCoarseMeshFolder, '*.vtk'))

    for iF, inMeshFile in tqdm.tqdm(enumerate(deformedMeshFiles )):

        inSparseCloud = coarseMeshFiles[iF]
        outInterpolatedObjFile = join(outFolder, Path(inMeshFile).stem + '.ply')
        M03_ToSparseFitting.interpolateWithSparsePointCloudSoftly(inMeshFile, inSparseCloud, outInterpolatedObjFile, inputs.skelDataFile,
                                             inputs.toSparsePCMat, laplacianMatFile=inputs.laplacianMatFile,
                                             softConstraintWeight=100,
                                             numRealCorners=1487, fixHandAndHead=False,
                                             )

    texture = r'texturemap_learned_LapW0.2_MaskTrue_L1.png'

    facesFile = 'FacesOnlySuit.json'
    facesOnSuit = set(json.load(open(facesFile)))
    # facesOnSuit = None

    ext = 'ply'
    withMtl = True
    rename = False
    plyFiles = glob.glob(join(outFolder, '*.' + ext))
    if rename:
        for plyF in plyFiles:
            fName = os.path.basename(plyF)
            if fName[0] != 'A':
                newFileName = join(outFolder, 'A'+fName)
                os.rename(plyF, newFileName)
                # print(plyF, newFileName)

    out_dir = os.path.join(outFolder, 'WithTextureCoord')
    converObjsInFolder(outFolder, out_dir, ext=ext, addA=False, withMtl=withMtl, textureFile=texture, facesOnSuit=facesOnSuit)

    objFilesToPly(out_dir, join(outFolder, 'PlyWithTextureCoord'))