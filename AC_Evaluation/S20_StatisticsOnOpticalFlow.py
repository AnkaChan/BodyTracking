from S06_2_RenderSilhouettesMultiFrames import *

from matplotlib import pyplot as plt
from Utility import *
from pathlib import Path
from os.path import join
import os
import cv2
import matplotlib

def computeOpticalFlowNorm(flow, sil):
    foregroundPixels = np.where(sil)
    flowMag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    avgFlowNorm = np.mean(flowMag[foregroundPixels])

    return avgFlowNorm

def readOpticalFlows(flowFiles, masks=None):
    flows = []
    if masks is None:
        for flowFile in flowFiles:
            flow = np.load(flowFile)
            flows.append(flow)
    else:
        for flowFile, mask in tqdm.tqdm(zip(flowFiles, masks)):
            flow = np.load(flowFile)
            flows.append(flow)

            # mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
            flow[np.where(np.logical_not(mask))] = 0
            masks.append(mask)
            # cv2.imshow('MaskedFlow', np.abs(flow[:,:,0])*10, )
            # cv2.waitKey()

    return flows

def getLocalizationErrors(errs):

    dis = errs

    # max
    statistics = {
        'max':np.max(dis),
        'mean':np.mean(dis),
        'median':np.median(dis),
        'p_95': np.percentile(dis, 95) , # return 50th percentile, e.g median.
        'p_99': np.percentile(dis, 99) , # return 50th percentile, e.g median.
        'p_999': np.percentile(dis, 99.9) , # return 50th percentile, e.g median.
        'p_9999': np.percentile(dis, 99.99) , # return 50th percentile, e.g median.
    }

    return dis, statistics

if __name__ == '__main__':
    coarseMeshFolder = r'F:\WorkingCopy2\2020_01_16_Lada_FinalAnimations\StandMotionNewPipeline_SegFiltered_3\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step100_Overlap0\Deformed'
    outSilhouetteFolder = r'X:\MocapProj\2021_01_02_OpticalFlowEvaluation\CoarseSilhouettes'
    outLBSToTPOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation\ToTrackingPoints\OpticalFlow'
    outInterpolateMeshOpticalFlowFolder = r'X:\MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/OpticalFlow'
    camParamF = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CameraParameters\cam_params.json'
    convertToM = True
    numFramesSelected = 100
    xlim = 15

    cfg = Config.RenderingCfg()
    cfg.sigma = 1e-10
    cfg.blurRange = 1e-10
    cfg.imgSize = 1080
    cfg.faces_per_pixel = 5

    cfg.numCams = 16
    cfg.batchSize = 8


    numZFill = 8
    frames = [str(i).zfill(numZFill) for i in range(8564, 8564 + numFramesSelected)]
    ext = 'obj'
    inTextureMeshFile = join(coarseMeshFolder,  'A' + frames[0] + '.' + ext)
    #First step: render silhouettes using the coarse mesh (only body part)
    # for frameName in tqdm.tqdm(frames):
    #     processRenderFolder = join(outSilhouetteFolder, frameName)
    #     meshFile = join(coarseMeshFolder,  'A' + frameName + '.' + ext)
    #     os.makedirs(processRenderFolder, exist_ok=True)
    #     M05_Visualization.renderFrame(meshFile, inTextureMeshFile, camParamF,
    #                                                   processRenderFolder,
    #                                                   cfg=cfg, convertToM=convertToM, rendererType='Silhouette')


    # outFinalMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/OpticalFlow'

    flowFilesLBSToTP = sortedGlob(join(outLBSToTPOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesInterpo = sortedGlob(join(outInterpolateMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))

    flowMaskFiles = [join(outSilhouetteFolder, frameName, 'AA' + frameName + '.png') for frameName in frames]

    flowFilesLBSToTP = flowFilesLBSToTP[:numFramesSelected]
    flowFilesInterpo = flowFilesInterpo[:numFramesSelected]

    # flowsLBS = readOpticalFlows(flowFilesLBSToTP, flowMaskFiles)
    masks = [cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE) for maskFile in flowMaskFiles]

    # flowsFinal = readOpticalFlows(flowFilesInterpo, masks)
    # flowsLBS = readOpticalFlows(flowFilesLBSToTP, masks)
    #
    # np.save('flowsFinal.npy', flowsFinal)
    # np.save('flowsLBS.npy', flowsLBS)
    flowsFinal = np.load('flowsFinal.npy')
    flowsLBS = np.load('flowsLBS.npy')


    avgNormsFinal = []
    errs = []
    for flow, mask in tqdm.tqdm(zip(flowsFinal, masks)):
        flowNorms = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

        errs.append(flowNorms[np.where(mask)].flatten())
        avgNormsFinal.append(np.mean(errs[-1]))

    errs = np.concatenate(errs)

    print("Average norms:", np.mean(errs))
    statistics = getLocalizationErrors(errs)
    print(statistics)

    font = {'family': 'normal',
            # 'weight': 'bold',
            'size': 20}

    matplotlib.rc('font', **font)
    kwargs = dict(alpha=0.5, bins=200, stacked=True)

    plt.figure('Flow Interpolated')
    # plt.hist(errsConsisRansacCutoff, **kwargs, color='r', label='With outlier filtering')
    plt.hist(errs, **kwargs)
    plt.xlim(0, xlim)
    plt.yscale('log')
    plt.gca().set(title='Norms of optical flow from interpolated mesh', ylabel='Errs')
    # plt.xlim(0, 75)
    plt.legend();
    # plt.savefig(join(outFolder, 'ReprojErrWithOutlierLessThan' + str(reprojErrCutoff)+'.png'), dpi=300)

    avgNormsLBS = []
    errs = []
    for flow, mask in tqdm.tqdm(zip(flowsLBS, masks)):
        flowNorms = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

        errs.append(flowNorms[np.where(mask)].flatten())
        avgNormsLBS.append(np.mean(errs[-1]))

    errs = np.concatenate(errs)

    plt.figure('Flow LBS')
    # plt.hist(errsConsisRansacCutoff, **kwargs, color='r', label='With outlier filtering')
    plt.hist(errs, **kwargs)
    plt.xlim(0, xlim)
    plt.yscale('log')
    plt.gca().set(title='Norms of optical flow from LBS deformation', ylabel='Errs')
    # plt.xlim(0, 75)
    plt.legend();
    # plt.savefig(join(outFolder, 'ReprojErrWithOutlierLessThan' + str(reprojErrCutoff)+'.png'), dpi=300)

    t = list(range(numFramesSelected))

    figure = plt.figure('CorrespondenceAccuracy')
    plt.plot(t, avgNormsLBS, label = 'Pure LBS ')
    # plt.plot(t, avgFlowMagsLBSToDense, label = 'Pure LBS - Keypoinits + OpenMVS Point Clouds')
    # plt.plot(t, avgFlowMagsIntpl, label = 'DTI - Keypoinits + Tracking Points')
    plt.plot(t, avgNormsFinal, label = 'Interpolated')
    plt.xlabel('Frames')
    plt.ylabel('Avg Optical Flow Norm')
    plt.legend()
    figure.set_size_inches(10, 4.5)
    plt.savefig('CorrespondenceAccuracy.png', dpi=300)

    plt.show()
    plt.waitforbuttonpress()