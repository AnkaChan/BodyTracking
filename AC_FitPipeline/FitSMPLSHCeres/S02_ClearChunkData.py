from Utility import *
import json
import pyvista as pv
from SkelFit.Data import *
if __name__ == '__main__':
    # inputRawChunkFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\GroundMotionNewPipeline_0_2000.clean.SegFit_03_inpaint_5_MaxMovement10.json'
    coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Ground\Deformed'
    # outChunkedFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\Lada_Ground.json'
    # animationExt = 'obj'

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Stand\Final_Smoothed1'
    # outChunkedFile = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Inputs\Lada_Stand.json'
    # animationExt = 'vtk'

    inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Katey_LongSeq\Final_Smoothed1'
    outChunkedFile = r'F:\WorkingCopy2\2020_11_26_SMPLSHFit\Inputs\Katey_Stand.json'
    animationExt = 'vtk'

    badVerts = getBadRestposeVerts(coarseSkelData)
    interval=None

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround\Final\Mesh'
    # coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
    # outChunkedFile = r'F:\WorkingCopy2\2020_11_11_TestSMPLSHCeresFit\SMPLSHTest_Ground.json'
    # convertToMM=True
    # badVerts = None
    # animationExt = 'ply'

    # frameData = json.load(open(inputRawChunkFile))
    pointCloudFilesToChunk(inputFinalAnimationFolder, outChunkedFile, interval=interval, discardedVerts=badVerts, convertToMM=False, inputExt=animationExt)



