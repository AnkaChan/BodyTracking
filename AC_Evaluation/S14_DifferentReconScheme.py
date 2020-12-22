import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
from Utility import *
import tqdm
import json
from SuitCapture import Triangulation, Data


if __name__ == '__main__':
    triangulateFolder = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\WholeSeq\TriangulationType1Only'

    testJsonFile = r'A02850.json'

    data = json.load(open(testJsonFile))

    print(data)

    cornerKeys, cornerKeysConfidence = Data.readProcessJsonFile(testJsonFile)

    labeler = Data.CornerLabeler()
    cornerUIds, cornerConf = labeler.labelCorners(cornerKeys, consistencyCheckScheme='maxConfidence', cornerKeysConfidence=cornerKeysConfidence)

    corr = labeler.cornerUIdsToCorrList(data['corners'], cornerUIds, cornerConf=cornerKeysConfidence)

    print(cornerUIds)