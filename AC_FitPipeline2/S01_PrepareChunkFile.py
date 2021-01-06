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

    inputFinalAnimationFolder = r'F:\WorkingCopy2\2020_01_22_FinalAnimations\Animations\Lada_Stand\Final_Smoothed1'
    outChunkedFile = r'F:\WorkingCopy2\2021_01_04_NewModelFitting\Inputs\Lada_Stand.json'
    coarseSkelData = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

    inExt = r'vtk'
    interval=None
    # interval=[0,20]
    numRealPts = 1487

    badVerts = getBadRestposeVerts(coarseSkelData)
    # interval=None
    pointCloudFilesToChunk(inputFinalAnimationFolder, outChunkedFile, interval=interval, discardedVerts=badVerts, convertToMM=False, inputExt=inExt, padTo=numRealPts)
