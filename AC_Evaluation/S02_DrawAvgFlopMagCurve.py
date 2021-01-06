from matplotlib import pyplot as plt
from Utility import *
from pathlib import Path
from os.path import join
import os
import cv2

def computeOpticalFlowNorm(flow, sil):
    foregroundPixels = np.where(sil)
    flowMag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    avgFlowNorm = np.mean(flowMag[foregroundPixels])

    return avgFlowNorm

if __name__ == '__main__':
    outLBSToTPOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation\ToTrackingPoints\OpticalFlow'
    outLBSToDenseOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation\ToDense\OpticalFlow'
    outInterpolateMeshOpticalFlowFolder = r'X:\MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/OpticalFlow'
    outFinalMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/OpticalFlow'

    flowFilesLBSToTP = sortedGlob(join(outLBSToTPOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesLBSToDense = sortedGlob(join(outLBSToDenseOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesInterpo = sortedGlob(join(outInterpolateMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesFinal = sortedGlob(join(outFinalMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))

    fittingFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation'
    ToKpAndDenseSilfolder = join(fittingFolder, 'ToDense', 'Silhouettes')
    ToTrackingPointsSilFolder = join(fittingFolder, 'ToTrackingPoints', 'Silhouettes')
    InterpolatedSilFolder = join(fittingFolder, 'Interpolated', 'Silhouettes')
    ImageBasedFittingSilFolder = join(fittingFolder, 'ImageBasedFitting', 'Silhouettes')


    # numFramesSelected = 120
    numFramesSelected = 100
    flowFilesLBSToTP = flowFilesLBSToTP[:numFramesSelected]
    flowFilesLBSToDense = flowFilesLBSToDense[:numFramesSelected]
    flowFilesInterpo = flowFilesInterpo[:numFramesSelected]
    flowFilesFinal = flowFilesFinal[:numFramesSelected]

    flowFilesFinalNormFile = join('Data', 'flowFilesFinal' + Path(flowFilesFinal[0]).stem + '_' + Path(flowFilesFinal[-1]).stem + '.npy')
    flowFilesLBSToTPNormFile = join('Data', 'flowFilesLBSToTP' + Path(flowFilesFinal[0]).stem + '_' + Path(flowFilesFinal[-1]).stem + '.npy')
    flowFilesLBSToDenseNormFile = join('Data', 'flowFilesLBSToDense' + Path(flowFilesFinal[0]).stem + '_' + Path(flowFilesFinal[-1]).stem + '.npy')
    flowFilesInterpoNormFile = join('Data', 'flowFilesInterpo' + Path(flowFilesFinal[0]).stem + '_' + Path(flowFilesFinal[-1]).stem + '.npy')

    avgFlowMagsFinal = []
    for flowFile in flowFilesFinal:
        flowlbs = np.load(flowFile)
        frameName = os.path.basename(flowFile)[:5]
        silFile = join(ImageBasedFittingSilFolder, frameName, 'A' + 'A' + frameName + '.png')
        sil = cv2.imread(silFile, cv2.IMREAD_GRAYSCALE)
        avgFlowMag = computeOpticalFlowNorm(flowlbs, sil)

        avgFlowMagsFinal.append(avgFlowMag)
    print('Avg Norm To Final: ', np.mean(avgFlowMagsFinal))
    np.save(flowFilesFinalNormFile, avgFlowMagsFinal)

    avgFlowMagsLBSToTP = []
    for flowFile in flowFilesLBSToTP:
        flowlbs = np.load(flowFile)
        frameName = os.path.basename(flowFile)[:5]
        silFile = join(ToTrackingPointsSilFolder, frameName, 'A' + 'A' + frameName + '.png')

        sil = cv2.imread(silFile, cv2.IMREAD_GRAYSCALE)

        # avgFlowMag = np.mean(flowMag)
        avgFlowMag =computeOpticalFlowNorm(flowlbs, sil)
        avgFlowMagsLBSToTP.append(avgFlowMag)

    print('Avg Norm To TP: ', np.mean(avgFlowMagsLBSToTP))
    np.save(flowFilesLBSToTPNormFile, avgFlowMagsLBSToTP)

    avgFlowMagsLBSToDense = []
    for flowFile in flowFilesLBSToDense:
        flowlbs = np.load(flowFile)

        frameName = os.path.basename(flowFile)[:5]
        silFile = join(ToKpAndDenseSilfolder, frameName, 'A' + 'A' + frameName + '.png')
        sil = cv2.imread(silFile, cv2.IMREAD_GRAYSCALE)
        avgFlowMag =computeOpticalFlowNorm(flowlbs, sil)

        avgFlowMagsLBSToDense.append(avgFlowMag)
    print('Avg Norm To Dense: ', np.mean(avgFlowMagsLBSToDense))
    np.save(flowFilesLBSToDenseNormFile, avgFlowMagsLBSToDense)

    avgFlowMagsIntpl = []
    for flowFile in flowFilesInterpo:
        flowlbs = np.load(flowFile)
        frameName = os.path.basename(flowFile)[:5]
        silFile = join(InterpolatedSilFolder, frameName, 'A' + 'A' + frameName + '.png')
        sil = cv2.imread(silFile, cv2.IMREAD_GRAYSCALE)
        avgFlowMag = computeOpticalFlowNorm(flowlbs, sil)

        avgFlowMagsIntpl.append(avgFlowMag)

    print('Avg Norm To Interpolated: ', np.mean(avgFlowMagsIntpl))
    np.save(flowFilesInterpoNormFile, avgFlowMagsIntpl)


    t = list(range(len(flowFilesLBSToTP)))

    figure = plt.figure('CorrespondenceAccuracy')
    plt.plot(t, avgFlowMagsLBSToTP, label = 'Pure LBS - Keypoinits + Tracking Points')
    # plt.plot(t, avgFlowMagsLBSToDense, label = 'Pure LBS - Keypoinits + OpenMVS Point Clouds')
    # plt.plot(t, avgFlowMagsIntpl, label = 'DTI - Keypoinits + Tracking Points')
    plt.plot(t, avgFlowMagsFinal, label = 'DTI + DTF - Keypoinits + Tracking Points')
    plt.xlabel('Frames')
    plt.ylabel('Avg Optical Flow Norm')
    plt.legend()
    figure.set_size_inches(10, 4.5)
    plt.savefig('CorrespondenceAccuracy.png', dpi=300)
    plt.show()
    plt.waitforbuttonpress()