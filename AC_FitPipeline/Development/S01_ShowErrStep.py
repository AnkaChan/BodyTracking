import sys, json
sys.path.insert(0,'../')
from Utility import *
import numpy as np
from matplotlib import pyplot as plt

def drawErrCurves(errs, outCurveFile):
    fig, a_loss = plt.subplots()
    a_loss.plot(errs, linewidth=3)
    a_loss.set_yscale('log')
    # a_loss.yscale('log')
    a_loss.set_title('losses: {}'.format(errs[-1]))
    a_loss.grid()
    fig.savefig(outCurveFile, dpi=256, transparent=False, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    # inParentFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse_NoInit'
    inParentFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    frameFolders = sortedGlob(join(inParentFolder, '*'))
    avgStep = 5

    for frameFolder in frameFolders:
        errFile = join(frameFolder, 'Errs.json')
        errs = json.load(open(errFile))

        stepSize = np.abs([errs[i] - errs[i + 1] for i in range(len(errs) - 1)])
        stepSizeAvg5 = [np.mean(np.abs(stepSize[i:i + avgStep])) for i in range(0, len(stepSize) - avgStep)]

        drawErrCurves(stepSize, join(frameFolder, 'ErrStep.png'))
        drawErrCurves(stepSizeAvg5, join(frameFolder, 'ErrStep' + str(avgStep) + '.png'))

