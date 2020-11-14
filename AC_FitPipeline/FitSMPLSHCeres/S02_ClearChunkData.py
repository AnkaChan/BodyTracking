from Utility import *
import json
import pyvista as pv
from SkelFit.Data import *
if __name__ == '__main__':
    # inputRawChunkFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\GroundMotionNewPipeline_0_2000.clean.SegFit_03_inpaint_5_MaxMovement10.json'
    inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Ground\Deformed'
    coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    outChunkedFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Lada_Ground.json'
    badVerts = getBadRestposeVerts(coarseSkelData)
    interval=None
    animationExt = 'obj'

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround\Final\Mesh'
    # coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    # outChunkedFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\SMPLSHTest_Ground.json'
    # convertToMM=True
    # badVerts = None
    # animationExt = 'ply'

    # frameData = json.load(open(inputRawChunkFile))
    pointCloudFilesToChunk(inputFinalAnimationFolder, outChunkedFile, interval=interval, discardedVerts=badVerts, convertToMM=False, inputExt=animationExt)



