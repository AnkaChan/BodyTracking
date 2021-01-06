from S17_EvaluateCornerdet import *

def getConfusionMatrixRejector(sess_rejector, posImgs, negImgs):
    rejector_logits = ia.run_rejector(sess_rejector, posImgs)
    qp_true_accepted = np.where(rejector_logits > 0)[0].shape[0]
    qp_false_rejected = np.where(rejector_logits <= 0)[0].shape[0]

    rejector_logits = ia.run_rejector(sess_rejector, negImgs)
    qp_false_accepted = np.where(rejector_logits > 0)[0].shape[0]
    qp_true_rejected = np.where(rejector_logits <= 0)[0].shape[0]

    return np.array([[qp_true_accepted, qp_false_accepted], [qp_false_rejected, qp_true_rejected]])




if __name__ == '__main__':
    dataFolder = r'E:\Dropbox\Mocap_Networks\cws_detector (1)\data_75'

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

    confusionMat = getConfusionMatrixRejector(sess_rejector, posImgs, negImgs)
    print('n=', posImgs.shape[0] + negImgs.shape[0], ', Confusion Matrix on Training set:\n', confusionMat)
    print('Confusion Matrix on Training set in percentage:\n', confusionMat * 100 /(posImgs.shape[0] + negImgs.shape[0]))

    posImgsTest = testImgs[np.where(testLabels)[0], :, :]
    negImgsTest = testImgs[np.where(np.logical_not(testLabels))[0], :, :]
    confusionMat = getConfusionMatrixRejector(sess_rejector, posImgsTest, negImgsTest)
    print('n=', posImgsTest.shape[0] + negImgsTest.shape[0], ', Confusion Matrix on Test set:\n', confusionMat)
    print('Confusion Matrix on Test set in percentage:\n', confusionMat * 100 /(posImgsTest.shape[0] + negImgsTest.shape[0]))

