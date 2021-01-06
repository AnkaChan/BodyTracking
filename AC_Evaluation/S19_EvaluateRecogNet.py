from S17_EvaluateCornerdet import *
suit_dict = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, 'A':7, 'B':8, 'C':9, 'D':10, 'E':11, 'F':12, 'G':13, 'J':14,
           'K':15, 'L':16, 'M':17, 'P':18, 'Q':19, 'R':20, 'T':21, 'U':22, 'V':23, 'Y':24}
inv_suit_dict = {v: k for k, v in suit_dict.items()}

def labelsToCode(recog_predictions1, recog_predictions2):
    code_strs = []
    for i in range(recog_predictions1.shape[0]):
        c1 = inv_suit_dict[recog_predictions1[i]]
        c2 = inv_suit_dict[recog_predictions2[i]]
        code_strs.append(str(c1) + str(c2))
    return code_strs

def recogAccuracy(predStrs, groundTruthStr):
    correctPredNum = 0
    for code1, code2 in zip(predStrs, groundTruthStr):
        if code1 == code2:
            correctPredNum += 1
    return correctPredNum / len(predStrs)

def runRecognizerBatched(sess_recognizer, data, batch_size = 6000):
    # recog_predictions1 = np.zeros((data.shape[0], 1))
    # recog_predictions1 = np.zeros((data.shape[0], 1))
    codes = []
    for b in range(0, data.shape[0], batch_size):
        b_end = min(b + batch_size, data.shape[0])
        recog_dict = {"recognizer/recognizer_imgs_ph:0": data[b:b_end,:,:,:], "recognizer/recognizer_pkeep_ph:0": 1.0}

        [recog_dens4a, recog_dens4b] = sess_recognizer.run(
            ["recognizer/recognizer_out_A:0", "recognizer/recognizer_out_B:0"], recog_dict)
        recog_predictions1Batch = np.argmax(recog_dens4a, 1)
        recog_predictions2Batch = np.argmax(recog_dens4b, 1)
        code_strs = []
        for i in range(recog_predictions1Batch.shape[0]):
            c1 = inv_suit_dict[recog_predictions1Batch[i]]
            c2 = inv_suit_dict[recog_predictions2Batch[i]]
            code_strs.append(str(c1) + str(c2))

        codes.extend(code_strs)
    return codes
if __name__ == '__main__':
    dataFolder = r'E:\Dropbox\Mocap_Networks\code_recog'
    recognizer_sess = join(quadpropDir, 'nets_recognizer/CNN_100_gen7_and_synth.ckpt')

    img_data = np.load(join(dataFolder, 'data_aug7/subImgSet.npy'))
    lab_data1 = np.load(join(dataFolder, 'data_aug7/labels1Set.npy'))
    lab_data2 = np.load(join(dataFolder, 'data_aug7/labels2Set.npy'))
    print(img_data.shape)
    print(lab_data1.shape)
    print(lab_data2.shape)

    synth_data = np.load(join(dataFolder, 'test_annot_04_synth/simgs.npy'))
    synth_labels = np.load(join(dataFolder, 'test_annot_04_synth/labels.npy'))
    print(synth_data.shape)
    print(synth_labels.shape)

    test_images = np.load(join(dataFolder, 'test_annot_02/simgs.npy'))
    test_labels = np.load(join(dataFolder, 'test_annot_02/labels.npy'))

    test2_images = np.load(join(dataFolder, 'test_annot_05/simgs.npy'))
    test2_labels = np.load(join(dataFolder, 'test_annot_05/labels.npy'))
    test_images = np.vstack([test_images, test2_images])
    test_labels = np.vstack([test_labels, test2_labels])

    saver3 = tf.train.import_meta_graph(recognizer_sess + '.meta', import_scope="recognizer")
    sess_recognizer = tf.Session()
    saver3.restore(sess_recognizer, recognizer_sess)


    trainStrs = labelsToCode(lab_data1, lab_data2)
    code_strs = runRecognizerBatched(sess_recognizer, img_data, )
    trainPredAcc = recogAccuracy(trainStrs, code_strs)

    print('Prediction accuracy on real training set:', trainPredAcc)

    synthStrs = labelsToCode(synth_labels[:,0], synth_labels[:,1])
    code_strs = runRecognizerBatched(sess_recognizer, synth_data, )
    testPredAcc = recogAccuracy(synthStrs , code_strs)
    print('Prediction accuracy on synthetic training set:', testPredAcc)

    testStrs = labelsToCode(test_labels[:,0], test_labels[:,1])
    code_strs = runRecognizerBatched(sess_recognizer, test_images, )
    testPredAcc = recogAccuracy(testStrs , code_strs)
    print('Prediction accuracy on test set:', testPredAcc)
