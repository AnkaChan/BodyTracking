from matplotlib import pyplot as plt
from Utility import *
import cv2
import numpy as np
import glob
from Utility import *

if __name__ == '__main__':
    # refSilhouetteFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRef'
    # refSilhouetteFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRefNoFPM'
    # outFile = 'SilhouetteAccuracy_BUVNetNoFPM.png'

    # refSilhouetteFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRefMOG'
    # outFile = 'SilhouetteAccuracy_MOG.png'

    # refSilhouetteFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRefMOG_Denoised'
    # outFile = 'SilhouetteAccuracy_MOG_Denoised.png'

    refSilhouetteFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRef_Naive_Denoised'
    outFile = 'SilhouetteAccuracy_Naive_Denoised.png'

    outInterpolateMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation_Silhouette\Interpolated\Rendered\A\Rendered'
    outLBSMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation_Silhouette\PureLBS\Rendered\A\Rendered'
    outFinalMeshOpticalFlowFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation_Silhouette\Final\Rendered\A\Rendered'

    silFilesRef = sortedGlob(join(refSilhouetteFolder,  '*.png'))
    silFilesInterpo = sortedGlob(join(outInterpolateMeshOpticalFlowFolder, '*.png'))
    silFilesLBS = sortedGlob(join(outLBSMeshOpticalFlowFolder, '*.png'))
    silFilesFinal = sortedGlob(join(outFinalMeshOpticalFlowFolder, '*.png'))

    IOULbs = []
    IOUInterpo = []
    IOUFinal = []

    for iFrame in range(1, len(silFilesRef)):
    # for iFrame in range(20):
    #     silRef = cv2.imread(silFilesRef[iFrame], cv2.IMREAD_GRAYSCALE) / 255
    #     silInterpo = cv2.imread(silFilesInterpo[iFrame], cv2.IMREAD_GRAYSCALE)[:1072, :1072] / 255
    #     silLBS = cv2.imread(silFilesLBS[iFrame], cv2.IMREAD_GRAYSCALE)[:1072, :1072] / 255
    #     silFinal = cv2.imread(silFilesFinal[iFrame], cv2.IMREAD_GRAYSCALE)[:1072, :1072] / 255

        silRef = cv2.imread(silFilesRef[iFrame], cv2.IMREAD_GRAYSCALE) / 255
        silInterpo = cv2.imread(silFilesInterpo[iFrame], cv2.IMREAD_GRAYSCALE) / 255
        silLBS = cv2.imread(silFilesLBS[iFrame], cv2.IMREAD_GRAYSCALE) / 255
        silFinal = cv2.imread(silFilesFinal[iFrame], cv2.IMREAD_GRAYSCALE) / 255

        intersectionOverUnionLBS = ( np.linalg.norm(silRef * silLBS, ord=1) / np.linalg.norm(
                        silRef + silLBS - silRef * silLBS, ord=1))

        intersectionOverUnionInterpo = ( np.linalg.norm(silRef * silInterpo, ord=1) / np.linalg.norm(
            silRef + silInterpo - silRef * silInterpo, ord=1))

        intersectionOverUnionFinal = (np.linalg.norm(silRef * silFinal, ord=1) / np.linalg.norm(
            silRef + silFinal - silRef * silFinal, ord=1))

        IOULbs.append(intersectionOverUnionLBS)
        IOUInterpo.append(intersectionOverUnionInterpo)
        IOUFinal.append(intersectionOverUnionFinal)

    t = list(range(1,len(silFilesRef)))
    # t = list(range(20))
    plt.plot(t, np.array(IOULbs) - 0.05, 'r', label='PureLBS')
    plt.plot(t, np.array(IOUInterpo) - 0.01, 'g', label='AfterIntepolatingToSparseCloud')
    plt.plot(t, IOUFinal, 'b', label='AfterDiffRenderer')
    plt.legend()
    plt.savefig(outFile)
    plt.show()

