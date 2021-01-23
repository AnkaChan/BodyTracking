import json, os
from Utility import *
from SkelFit.Data import *

if __name__ == '__main__':
    # os.makedirs(r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs', exist_ok=True)
    #
    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Ground\Final_Smoothed1'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Stand.json'
    # coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # inputFinalAnimationFolder = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487\BackToRestpose'
    # outChunkedFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\2020_12_27_betterCoarseMesh\Mesh1487\BackToRestpose\Restpose.json'
    # coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Stand\Final_Smoothed1'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Stand.json'
    # coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Katey_LongSeq\Final_Smoothed1'

    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Stand.json'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Stand_1Frame.json'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_CalibrationSeqs.json'
    coarseSkelData = r'..\Data\2020_12_27_betterCoarseMesh\Mesh1487_Katey\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'

    # inExt = r'vtk'
    # inExt = r'obj'
    # interval=None
    # interval=[54,54+1500]
    # interval=[54+1299,54+1300]
    # interval=[54,54+6000]

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose\Triangulation_RThres1.5_HardRThres_1.5'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Rolling.json'
    # interval=[3440, 3762] # start from 0

    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose\Triangulation_RThres1.5_HardRThres_1.5'
    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2021_01_14_AnimatinoSeqs\HolelyMeshes\Katey_SquatSpin\Threshold_0inpaint_5_MaxMovement15'
    # inExt = r'ply'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_SquatSpin.json'
    # # interval=[9196 - 3446, 9495 - 3446] # start from 0
    # interval=None # start from 0

    inputFinalAnimationFolder = r'F:\WorkingCopy2\2021_01_14_AnimatinoSeqs\HolelyMeshes\Katey_Interpolation\Threshold_0inpaint_5_MaxMovement15'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Ballet.json'
    # interval=[8868 - 3446, 8869 - 3446] # start from 0
    outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Katey_Interpolation_ReduceFlickering.json'
    # interval=[4799 - 3446, 6000 - 3446] # start from 0
    interval=None # start from 0
    inExt = r'ply'


    # inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\TriangulationType1Only'
    # outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Yoga.json'
    #
    # interval=[9613,  10142] # start from 0

    numRealPts = 1487

    badVerts = getBadRestposeVerts(coarseSkelData)
    badVerts = badVerts[np.where(badVerts<numRealPts)]
    # interval=None
    pointCloudFilesToChunk(inputFinalAnimationFolder, outChunkedFile, interval=interval, discardedVerts=badVerts, convertToMM=False, inputExt=inExt, padTo=numRealPts)
