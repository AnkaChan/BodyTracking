from matplotlib import pyplot as plt
from Utility import *

if __name__ == '__main__':
    outLBSMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/PureLBS/OpticalFlow'
    outInterpolateMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/OpticalFlow'
    outFinalMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/OpticalFlow'

    flowFilesLBS = sortedGlob(join(outLBSMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesInterpo = sortedGlob(join(outInterpolateMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))
    flowFilesFinal = sortedGlob(join(outFinalMeshOpticalFlowFolder, 'Flow', 'A', '*.npy'))

    numFramesSelected = 120
    flowFilesLBS = flowFilesLBS[:numFramesSelected]
    flowFilesInterpo = flowFilesInterpo[:numFramesSelected]
    flowFilesFinal = flowFilesFinal[:numFramesSelected]

    avgFlowMagsLBS = []
    for flowFile in flowFilesLBS:
        flowlbs = np.load(flowFile)
        flowMag = np.sqrt(flowlbs[:,:,0]**2 + flowlbs[:,:,1]**2)

        avgFlowMag = np.mean(flowMag)
        avgFlowMagsLBS.append(avgFlowMag)

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

    t = list(range(len(flowFilesLBS)))

    plt.plot(t, avgFlowMagsLBS, 'r')
    plt.plot(t, avgFlowMagsIntpl, 'g')
    plt.plot(t, avgFlowMagsFinal, 'b')
    plt.show()