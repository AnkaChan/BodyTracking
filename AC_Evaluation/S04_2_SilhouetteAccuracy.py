from matplotlib import pyplot as plt
from Utility import *
import cv2
import numpy as np
import glob
from Utility import *
import tqdm
if __name__ == '__main__':
    fittingFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation'

    ToKpAndDensefolder = join(fittingFolder, 'ToDense', 'Silhouettes')
    ToTrackingPointsFolder = join(fittingFolder, 'ToTrackingPoints', 'Silhouettes')
    InterpolatedFolder = join(fittingFolder, 'Interpolated', 'Silhouettes')
    ImageBasedFittingFolder = join(fittingFolder, 'ImageBasedFitting', 'Silhouettes')
    silhouettesFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Silhouettes'

    # frames = [str(i) for i in range(10459, 10459 + 50)]
    # frames = [str(i) for i in range(10459, 10459 + 230)]
    # frames = [str(i) for i in range(10459, 10459 + 200)]
    frames = [str(i) for i in range(10459, 10459 + 10)]

    folders = [ ToTrackingPointsFolder, ToKpAndDensefolder, InterpolatedFolder, ImageBasedFittingFolder]

    IOUToTP = []
    IOUToDense = []
    IOUInterpo = []
    IOUFinal = []

    statistics = [IOUToTP, IOUToDense, IOUInterpo, IOUFinal]
    # Gen = False
    Gen = True

    dataNames = ['LBSToTrackingPoints_'+ frames[0] + '_' + frames[-1] +'.npy',
                 'LBSToDense_' + frames[0] + '_' + frames[-1] + '.npy',
                 'Interpolated_' + frames[0] + '_' + frames[-1] + '.npy',
                 'AfterDiffRenderer' + frames[0] + '_' + frames[-1] + '.npy',
                 ]

    if Gen:
        for frameName in tqdm.tqdm(frames):
            refSilsFodler = join(silhouettesFolder, frameName, 'FinalSils')
            refSilFiles = sortedGlob(join(refSilsFodler, '*.png'))

            refImgs = [cv2.imread(refF, cv2.IMREAD_GRAYSCALE)/255 for refF in refSilFiles]

            for folder, IOUList in zip(folders, statistics):
                renderedSilsFolder = join(folder, frameName )
                renderedSilsFs = sortedGlob(join(renderedSilsFolder, '*.png'))
                renderedSils = [cv2.imread(renderedF, cv2.IMREAD_GRAYSCALE)/255 for renderedF in renderedSilsFs]

                intersectionOverUnionLBS = 0
                for renderedSil, refSil in zip(renderedSils, refImgs):

                    intersectionOverUnionLBS += ( np.linalg.norm(renderedSil * refSil, ord=1) / np.linalg.norm(
                                    renderedSil + refSil - renderedSil * refSil, ord=1))

                intersectionOverUnionLBS = intersectionOverUnionLBS / len(renderedSils)
                IOUList.append(intersectionOverUnionLBS)

        dataOutFolder = 'Data'

        np.save(join(dataOutFolder, dataNames[0]), np.array(IOUToTP), )
        np.save(join(dataOutFolder, dataNames[1]), np.array(IOUToDense), )
        np.save(join(dataOutFolder, dataNames[2]), np.array(IOUInterpo), )
        np.save(join(dataOutFolder, dataNames[3]), np.array(IOUFinal), )
    else:
        dataOutFolder = 'Data'
        IOUToTP = np.load(join(dataOutFolder, dataNames[0]))
        IOUToDense = np.load(join(dataOutFolder, dataNames[1]) )
        IOUInterpo = np.load(join(dataOutFolder, dataNames[2]))
        IOUFinal = np.load(join(dataOutFolder, dataNames[3]), )

    t = list(range(0,len(frames)))
    # t = list(range(20))
    plt.plot(t, np.array(IOUToTP), label='Pure LBS - Keypoinits + Tracking Points')
    plt.plot(t, np.array(IOUToDense), label='Pure LBS   - Keypoinits + OpenMVS Point Clouds ')
    plt.plot(t, np.array(IOUInterpo), label='DTI - Keypoinits + Tracking Points')
    plt.plot(t, np.array(IOUFinal), label='DTI + DTF - Keypoinits + Tracking Points')
    plt.legend()
    plt.savefig('SillouettesAccuracy.pdf')
    plt.show()
    plt.waitforbuttonpress()

