quadpropDir = r'C:\Code\MyRepo\ChbCapture\08_CNNs\QuadProposals'
import sys
sys.path.insert(0, quadpropDir)
import imganalysis as ia
import quadprops as qps
import tensorflow as tf
from os.path import join
import numpy as np
import os

from matplotlib import pyplot as plt

def confusionMatrix(labels, pred):
    numData = labels.shape[0]

    posLabels = np.where((labels[:, 0:1]))
    negLabels = np.where(np.logical_not(labels[:, 0:1]))

    truePostive = np.where(labels[posLabels]==pred[posLabels])[0]
    falseNeg =  np.where(labels[posLabels]!=pred[posLabels])[0]

    trueNeg = np.where(labels[negLabels]==pred[negLabels])[0]
    falsePos =  np.where(labels[negLabels]!=pred[negLabels])[0]

    confuMat = np.array([[len(truePostive), len(falsePos)], [len(falseNeg), len(trueNeg)]])

    return numData, confuMat

def getConfusionMat(cornerdetSess, imgs, label):
    (d4a, d4b) = ia.cornerdet_inference(cornerdetSess, imgs, oldInterface=True)

    d4a[np.where(d4a > confidence_threshold)] = 1
    d4a[np.where(d4a < confidence_threshold)] = 0

    numData, confuMat = confusionMatrix(label[:, 0:1], d4a)

    return numData, confuMat, d4b

def getLocalizationErrors(localizationPred, labels, denormalizationScale=8):
    posLabels = np.where((labels[:, 0:1]))[0]

    diff = localizationPred[posLabels, :] - labels[posLabels, 1:]
    dis = np.sqrt(diff[:, 0]**2, diff[:, 1]**2) * denormalizationScale

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
    inDataFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\data_28'
    # cornerdet_sess = join(quadpropDir, 'CornerDectector2/20200105_13h57m.ckpt') # type 2 corner detector
    cornerdet_sess = join(quadpropDir, 'nets', '28_renamed.ckpt')
    outFolder = 'output/S17_EvaluateCornerdet'

    os.makedirs(outFolder, exist_ok=True)

    confidence_threshold = -2

    trainingImgs = np.load(join(inDataFolder, 'train_imgs.npy'))
    trainingLabels = np.load(join(inDataFolder, 'train_labels.npy'))
    testgImgs = np.load(join(inDataFolder, 'test_imgs.npy'))
    testLabels = np.load(join(inDataFolder, 'test_labels.npy'))

    tf.reset_default_graph()

    saver = tf.train.import_meta_graph(cornerdet_sess + '.meta', import_scope="cornerdet")
    sess_cornerdet = tf.Session()
    saver.restore(sess_cornerdet, cornerdet_sess)

    # crops, i_list, j_list = qps.gen_crops()
    np.set_printoptions(precision=3, suppress=True)

    numDataTrain, confuMatTrain, d4bTrain = getConfusionMat(sess_cornerdet, trainingImgs, trainingLabels)
    print('n=', trainingImgs.shape[0], ', ConfuMat on train:\n', confuMatTrain)
    print('ConfuMat in percentage on train:\n', confuMatTrain*100/numDataTrain)

    # print(len(np.where(trainingLabels[:, 0:1]!=d4a)[0])/trainingLabels.shape[0])

    numDataTest, confuMatTest, d4bTest = getConfusionMat(sess_cornerdet, testgImgs, testLabels)
    print('n=', testgImgs.shape[0], ', ConfuMat on test:\n', confuMatTest)
    print('ConfuMat in percentage on test:\n', confuMatTest*100/numDataTest)
    # print(len(np.where(trainingLabels[:, 0:1]!=d4a)[0])/trainingLabels.shape[0])

    # evaluate localization Errors
    truePostive = trainingLabels
    trainLocalizationErrs, statisticsTrain = getLocalizationErrors(d4bTrain, trainingLabels)
    print('statisticsTrain\n', statisticsTrain)

    testLocalizationErrs, statisticsTest = getLocalizationErrors(d4bTest, testLabels)
    print('statisticsTest\n', statisticsTest)

    # draw figures:
    fig, ax = plt.subplots()
    ax.hist(trainLocalizationErrs, 100, )
    ax.set_xlabel('Localization errors (in pixels)')
    ax.set_ylabel('Numbers')
    ax.set_yscale('log')
    ax.set_title('Localization errors for Cornerdet on Training Set')

    fig.savefig(join(outFolder, 'TrainLocalizationErrs.png'), dpi=400, bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    ax.hist(testLocalizationErrs, 100, )
    ax.set_xlabel('Localization errors (in pixels)')
    ax.set_ylabel('Numbers')
    ax.set_yscale('log')
    ax.set_title('Localization errors for Cornerdet on Test Set')

    fig.savefig(join(outFolder, 'TestLocalizationErrs.png'), dpi=400, bbox_inches='tight', pad_inches=0)

    plt.show()
    # plt.waitforbuttonpress()