import numpy as np
from matplotlib import pyplot as plt
import json, os


if __name__ == '__main__':
    inPoseFittingErrsFile = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\ErrorAnalysis\ErrsPose.json'
    inPerVertFittingErrsFile = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\ErrorAnalysis\ErrsPerVertex.json'

    avgStep = 10

    errs = json.load(open(inPoseFittingErrsFile))

    imageLoss = errs['ImageLoss']

    # plt.figure('ImageLoss_Pose')
    # plt.plot(imageLoss)


    stepSize = np.abs([imageLoss[i] - imageLoss[i+1] for i in range(len(imageLoss)-1)])
    stepSizeAvg5 = [np.mean(np.abs(stepSize[i:i+avgStep])) for i in range(0, len(stepSize)-avgStep)]
    # plt.figure('ImageLoss_step_Pose')
    # plt.plot(stepSize)
    plt.figure('ImageLoss_step_Pose_avg' + str(avgStep))
    plt.plot(stepSizeAvg5)

    errsPerVert = json.load(open(inPerVertFittingErrsFile))
    imageLoss = errsPerVert['ImageLoss']

    # plt.figure('ImageLoss_PerVert')
    # plt.plot(imageLoss)

    stepSize = np.abs([imageLoss[i] - imageLoss[i+1] for i in range(len(imageLoss)-1)])

    stepSizeAvg5 = [np.mean(np.abs(stepSize[i:i+avgStep])) for i in range(0, len(stepSize)-avgStep)]

    # plt.figure('ImageLoss_step_PerVert')
    # plt.plot(stepSize)
    plt.figure('ImageLoss_step_PerVert_avg' + str(avgStep))
    plt.plot(stepSizeAvg5)

    plt.show()
    plt.waitforbuttonpress( )
