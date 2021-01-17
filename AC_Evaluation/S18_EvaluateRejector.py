from S17_EvaluateCornerdet import *
import cv2

def getConfusionMatrixRejector(sess_rejector, posImgs, negImgs):
    rejector_logits = ia.run_rejector(sess_rejector, posImgs)
    qp_true_accepted = np.where(rejector_logits > 0)[0].shape[0]
    qp_false_rejected = np.where(rejector_logits <= 0)[0].shape[0]

    falseNegExamples = posImgs[np.where(rejector_logits <= 0)[0], :, :, :]

    rejector_logits = ia.run_rejector(sess_rejector, negImgs)
    qp_false_accepted = np.where(rejector_logits > 0)[0].shape[0]
    qp_true_rejected = np.where(rejector_logits <= 0)[0].shape[0]

    falsePositiveExamples = negImgs[np.where(rejector_logits > 0)[0], :, :, :]
    return np.array([[qp_true_accepted, qp_false_accepted], [qp_false_rejected, qp_true_rejected]]), falseNegExamples, falsePositiveExamples




if __name__ == '__main__':
    dataFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\data_75'
    outFolder = join('output', 'S28_EvaluateRejector')
    os.makedirs(outFolder, exist_ok=True)

    outFolderFalseNeg = join(outFolder, 'FalseNegtives')
    outFolderFalsePos = join(outFolder, 'FalsePositives')
    os.makedirs(outFolderFalseNeg, exist_ok=True)
    os.makedirs(outFolderFalsePos, exist_ok=True)

    posImgs = np.load(join(dataFolder, 'pos_augmented.npy'))
    negImgs = np.load(join(dataFolder, 'neg_augmented.npy'))

    testImgs = np.load(join(dataFolder, 'test_imgs.npy'))
    testLabels = np.load(join(dataFolder, 'test_labels.npy'))

    # prepare the rejector net
    rejector_sess = join(quadpropDir, 'Rejector2/20200114_14h07m.ckpt')
    np.set_printoptions(precision=3, suppress=True)

    saver2 = tf.train.import_meta_graph(rejector_sess + '.meta', import_scope="rejector")
    sess_rejector = tf.Session()
    saver2.restore(sess_rejector, rejector_sess)

    confusionMat, falseNegImgs, falsePosImgs = getConfusionMatrixRejector(sess_rejector, posImgs, negImgs)
    print('n=', posImgs.shape[0] + negImgs.shape[0], ', Confusion Matrix on Training set:\n', confusionMat)
    print('Confusion Matrix on Training set in percentage:\n', confusionMat * 100 /(posImgs.shape[0] + negImgs.shape[0]))

    for i in range(falseNegImgs.shape[0]):
        cv2.imwrite(join(outFolderFalseNeg, str(i).zfill(5) + '.png'), falseNegImgs[i, ...])
    for i in range(falsePosImgs.shape[0]):
        cv2.imwrite(join(outFolderFalsePos, str(i).zfill(5) + '.png'), falsePosImgs[i, ...])

    posImgsTest = testImgs[np.where(testLabels)[0], :, :]
    negImgsTest = testImgs[np.where(np.logical_not(testLabels))[0], :, :]
    confusionMat, falseNegImgs, falsePosImgs = getConfusionMatrixRejector(sess_rejector, posImgsTest, negImgsTest)
    print('n=', posImgsTest.shape[0] + negImgsTest.shape[0], ', Confusion Matrix on Test set:\n', confusionMat)
    print('Confusion Matrix on Test set in percentage:\n', confusionMat * 100 /(posImgsTest.shape[0] + negImgsTest.shape[0]))


    for i in range(falseNegImgs.shape[0]):
        cv2.imwrite(join(outFolderFalseNeg, 'Test'+ str(i).zfill(5) + '.png'), falseNegImgs[i, ...])
    for i in range(falsePosImgs.shape[0]):
        cv2.imwrite(join(outFolderFalsePos, 'Test'+ str(i).zfill(5) + '.png'), falsePosImgs[i, ...])