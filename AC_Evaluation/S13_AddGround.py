from SkelFit.Visualization import *
from SkelFit.Data import *
from SkelFit.Geometry import *
from SkelFit.Model import *


if __name__ == '__main__':
    # inFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Interpolated\Katey_LongSeq'
    # fittingToVtk(inFolder, extName= 'ply', addGround=True, meshWithFaces=None)

    # inFolder = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Video\Katey\TriangulationType1Only\TriangulationType1Only'
    # fittingToVtk(inFolder, extName= 'obj', addGround=True, meshWithFaces=None)

    inFolder = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\StandMotion2Cut\Triangulation'
    fittingToVtk(inFolder, extName= 'obj', addGround=False, meshWithFaces=None)
    pass