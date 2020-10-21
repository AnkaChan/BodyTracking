from matplotlib import pyplot as plt
from Utility import *

if __name__ == '__main__':
    outLBSToTPOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation\ToTrackingPoints\OpticalFlow'
    outLBSToDenseOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation\ToDense\OpticalFlow'
    outInterpolateMeshOpticalFlowFolder = r'X:\MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/OpticalFlow'
    outFinalMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/OpticalFlow'

    flowFilesLBSToTP = sortedGlob(join(outLBSToTPOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesLBSToDense = sortedGlob(join(outLBSToDenseOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesInterpo = sortedGlob(join(outInterpolateMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesFinal = sortedGlob(join(outFinalMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))

    # numFramesSelected = 120
    numFramesSelected = 100
    flowFilesLBSToTP = flowFilesLBSToTP[:numFramesSelected]
    flowFilesLBSToDense = flowFilesLBSToDense[:numFramesSelected]
    flowFilesInterpo = flowFilesInterpo[:numFramesSelected]
    flowFilesFinal = flowFilesFinal[:numFramesSelected]

    avgFlowMagsLBSToTP = []
    for flowFile in flowFilesLBSToTP:
        flowlbs = np.load(flowFile)
        flowMag = np.sqrt(flowlbs[:,:,0]**2 + flowlbs[:,:,1]**2)

        avgFlowMag = np.mean(flowMag)
        avgFlowMagsLBSToTP.append(avgFlowMag)

    avgFlowMagsLBSToDense = []
    for flowFile in flowFilesLBSToDense:
        flowlbs = np.load(flowFile)
        flowMag = np.sqrt(flowlbs[:, :, 0] ** 2 + flowlbs[:, :, 1] ** 2)

        avgFlowMag = np.mean(flowMag)
        avgFlowMagsLBSToDense.append(avgFlowMag)

    avgFlowMagsIntpl = []
    for flowFile in flowFilesInterpo:
        flowlbs = np.load(flowFile)
        flowMag = np.sqrt(flowlbs[:, :, 0] ** 2 + flowlbs[:, :, 1] ** 2)

        avgFlowMag = np.mean(flowMag)
        avgFlowMagsIntpl.append(avgFlowMag)

    avgFlowMagsFinal = []
    for flowFile in flowFilesFinal:
        flowlbs = np.load(flowFile)
        flowMag = np.sqrt(flowlbs[:, :, 0] ** 2 + flowlbs[:, :, 1] ** 2)

        avgFlowMag = np.mean(flowMag)
        avgFlowMagsFinal.append(avgFlowMag)

    t = list(range(len(flowFilesLBSToTP)))

    plt.plot(t, avgFlowMagsLBSToTP, label = 'Pure LBS - Keypoinits + Tracking Points')
    plt.plot(t, avgFlowMagsLBSToDense, label = 'Pure LBS - Keypoinits + OpenMVS Point Clouds')
    plt.plot(t, avgFlowMagsIntpl, label = 'DTI - Keypoinits + Tracking Points')
    plt.plot(t, avgFlowMagsFinal, label = 'DTI + DTF - Keypoinits + Tracking Points')
    plt.legend()
    plt.savefig('CorrespondenceAccuracy.pdf')
    plt.show()
    plt.waitforbuttonpress()