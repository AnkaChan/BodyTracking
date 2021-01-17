import numpy as np
from matplotlib import  pyplot as plt
import matplotlib
from Utility import *
import tqdm
import json
from SuitCapture import Triangulation, Data, Camera
from pathlib import Path
def write_obj(verts, file_name):
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

def reconstructFromCorrs(corrs, camProjMats, scheme='DLT'):
    if len(corrs) == 0:
        return []
    numCorners = len(corrs[0])
    triangulations = []
    errors = []
    for iC in range(numCorners):
        camPts = []
        selectedCamProjMats = []
        for iCam in range(len(camProjMats)):
            if iC < len(corrs[iCam]):
                # camPts.append(corrs[iCam][i])
                # confidence.append(corrs[iCam][i][2])
                # selectedCamProjMats.append(camProjMats[iCam])
                if corrs[iCam][iC][0] != -1:
                    camPts.append(corrs[iCam][iC])
                    selectedCamProjMats.append(camProjMats[iCam])

        if len(camPts)>=2:
            if scheme == 'DLT':
                X, errs = Triangulation.mulCamsDLT(camPts, selectedCamProjMats)
            elif scheme == 'RANSAC':
                X, errs = Triangulation.mulCamsRansac(camPts, selectedCamProjMats)
        else:
            X = [0,0,-1]
            errs = np.array([0])

        triangulations.append(X)
        errors.append(errs)

    return triangulations, errors


if __name__ == '__main__':
    # triangulateFolder = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\WholeSeq\TriangulationType1Only'
    #
    # testJsonFile = r'A02850.json'
    #
    # data = json.load(open(testJsonFile))
    #
    # print(data)
    #
    # cornerKeys, cornerKeysConfidence = Data.readProcessJsonFile(testJsonFile)
    #
    # labeler = Data.CornerLabeler()
    # cornerUIds, cornerConf = labeler.labelCorners(cornerKeys, consistencyCheckScheme='maxConfidence', cornerKeysConfidence=cornerKeysConfidence)
    #
    # corr = labeler.cornerUIdsToCorrList(data['corners'], cornerUIds, cornerConf=cornerKeysConfidence)
    #
    # print(cornerUIds)

    processFolder = r'Z:\2019_12_13_Lada_Capture\Converted'
    outputFolder = r'F:\WorkingCopy2\2020_12_22_ReconstructionEvaluation'
    calibrationDataFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CameraParameters\cam_params.json'
    corrFolder = join(outputFolder, 'Corrs')
    reconFolder = join(outputFolder, 'Recon')
    processName = 'Pattern_quad_proposal_2850_14501'
    camFolders = sortedGlob(join(processFolder, '*'))

    camNames = [Path(camFolder).stem for camFolder in camFolders]

    # frameNames = [str(iF).zfill(5) for iF in range(2850, 2850+10000)]
    frameNames = [str(iF).zfill(5) for iF in range(2850+15, 2850+10000)]

    labeler = Data.CornerLabeler()

    # configurate cameras
    camParams, _ = Camera.loadCamParams(calibrationDataFile)
    camProjMats = []
    for iCam in range(len(camParams)):
        camParam = camParams[iCam]
        I, E = Camera.calibrationParamsToIEMats(camParam, True)

        projMat = I @ E
        # pts2D = Triangulation.projectPoints(mesh.points, projMat)
        # pts2Ds.append(pts2D)
        camProjMats.append(projMat)

    os.makedirs(corrFolder, exist_ok=True)
    corrFolderWithConsistencyCheck = join(corrFolder, 'WithConsis')
    os.makedirs(corrFolderWithConsistencyCheck, exist_ok=True)
    corrFolderWithoutConsistencyCheck = join(corrFolder, 'WithoutConsis')
    os.makedirs(corrFolderWithoutConsistencyCheck, exist_ok=True)

    os.makedirs(reconFolder, exist_ok=True)
    reconFolderWithoutConsis = join(reconFolder, 'WithoutConsis')
    os.makedirs(reconFolderWithoutConsis, exist_ok=True)
    reconFolderWithConsis = join(reconFolder, 'WithConsis')
    os.makedirs(reconFolderWithConsis, exist_ok=True)
    reconFolderWithConsisRANSAC = join(reconFolder, 'WithConsisRANSAC')
    os.makedirs(reconFolderWithConsisRANSAC, exist_ok=True)

    # iterate all the frame names
    for frameName in tqdm.tqdm(frameNames):
        # gen processed json for each camera
        processedJsons = [join(camFolder, processName, camName+frameName+'.json') for camFolder, camName in zip(camFolders, camNames)]

        # read json, convert to corner keys
        corrsWithConsis = []
        corrsWithoutConsis = []
        for jFile in processedJsons:
            corners, cornerKeys, cornerKeysConfidence = Data.readProcessJsonFile(jFile)

            # corner keys to uIds & corrs without consistency check
            cornerUIds, cornerConf = labeler.labelCorners(cornerKeys, consistencyCheckScheme='maxConfidence', cornerKeysConfidence=cornerKeysConfidence)
            corr = labeler.cornerUIdsToCorrList(corners, cornerUIds, cornerConf=cornerConf)
            corrsWithoutConsis.append(corr)

            # corner keys to uIds & corrs with consistency check
            cornerUIds = labeler.labelCorners(cornerKeys, consistencyCheckScheme='discard', cornerKeysConfidence=cornerKeysConfidence)
            corr = labeler.cornerUIdsToCorrList(corners, cornerUIds, cornerConf=cornerConf)
            corrsWithConsis.append(corr)

        #   save corrs
        json.dump(corrsWithConsis, open(join(corrFolderWithConsistencyCheck, frameName+'.json'), 'w'))
        json.dump(corrsWithoutConsis, open(join(corrFolderWithoutConsistencyCheck, frameName+'.json'), 'w'))

        # reconstruct directly on corrs without consistency check
        triangulationsWithoutConsis, errorsWithoutConsis = reconstructFromCorrs(corrsWithoutConsis, camProjMats, scheme='DLT')
        write_obj(triangulationsWithoutConsis, join(reconFolderWithoutConsis, frameName+'.obj'))
        json.dump([err.tolist() for err in errorsWithoutConsis], open(join(reconFolderWithoutConsis, frameName+'Errs.json'), 'w'))

        # reconstruct directly on corrs with consistency check
        triangulationsWithConsis, errorsWithConsis = reconstructFromCorrs(corrsWithConsis, camProjMats, scheme='DLT')
        write_obj(triangulationsWithConsis, join(reconFolderWithConsis, frameName+'.obj'))
        json.dump([err.tolist() for err in errorsWithConsis], open(join(reconFolderWithConsis, frameName+'Errs.json'), 'w'))

        # reconstruct directly on corrs with consistency check and RANSAC
        triangulationsWithConsisRansac, errorsWithConsisRansac = reconstructFromCorrs(corrsWithConsis, camProjMats, scheme='RANSAC')
        write_obj(triangulationsWithConsisRansac, join(reconFolderWithConsisRANSAC, frameName+'.obj'))
        json.dump([err.tolist() for err in errorsWithConsisRansac], open(join(reconFolderWithConsisRANSAC, frameName+'Errs.json'), 'w'))