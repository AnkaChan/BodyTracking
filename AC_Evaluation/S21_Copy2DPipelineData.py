from Utility import *
import json

from quadprops import *
from SuitCapture import Triangulation, Data, Camera


if __name__ == '__main__':
    inDataFolder = r'Z:\2020_01_01_KateyCapture\Converted'
    interval = [3500, 5000]
    processName = r'Pattern_quad_proposal_3446_20000'

    camFolders = sortedGlob(join(inDataFolder, '*'))
    recogJsonData = r'H03446.json'
    jsonData = json.load(open(recogJsonData))
    print(jsonData)

    # corner detector result: corners
    # candidate quads: accpted
    # valid candidate quads: accept_qi
    # recognizer result: ciode
    # all the candidates

    # quad_proposals()
    # cornerUIds = labeler.labelCorners(cornerKeys, consistencyCheckScheme='discard',
    #                                   cornerKeysConfidence=cornerKeysConfidence)
    # corners, cornerKeys, cornerKeysConfidence = Data.readProcessJsonFile(jFile)