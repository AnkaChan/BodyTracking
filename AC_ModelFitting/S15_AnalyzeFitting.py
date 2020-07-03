import numpy as np
from matplotlib import pyplot as plt
import json, os


if __name__ == '__main__':
    inPoseFittingErrsFile = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\ErrorAnalysis\ErrsPose.json'
    errs = json.load(open(inPoseFittingErrsFile))

    imageLoss = errs['ImageLoss']

    plt.figure('ImageLoss_Pose')
    plt.plot(imageLoss)

    stepSize = [imageLoss[i] - imageLoss[i+1] for i in range(len(imageLoss)-1)]

    plt.figure('ImageLoss_step_Pose')
    plt.plot(stepSize)
    plt.show()

    plt.waitforbuttonpress()