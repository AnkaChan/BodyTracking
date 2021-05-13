import cv2
import pytesseract
from Utility import *

suit_dict = {'1':0, '2':1, '3':2, '4':3, '5':4, '6':5, '7':6, 'A':7, 'B':8, 'C':9, 'D':10, 'E':11, 'F':12, 'G':13, 'J':14,
           'K':15, 'L':16, 'M':17, 'P':18, 'Q':19, 'R':20, 'T':21, 'U':22, 'V':23, 'Y':24}
inv_suit_dict = {v: k for k, v in suit_dict.items()}

if __name__ == '__main__':

    outFolder = r'output\S36_Ablation_Recognizer'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    # test set 2
    # test_images = np.load(r'NewSuit/TestData/imgs_RecogNet.npy')
    # test_labels = np.load(r'NewSuit/TestData/labels_RecogNet.npy')

    # test set 1
    dataFolder = r'E:\Dropbox\Mocap_Networks\code_recog'

    test_images = np.load(join(dataFolder, 'test_annot_02/simgs.npy'))
    test_labels = np.load(join(dataFolder, 'test_annot_02/labels.npy'))

    # img = cv2.imread(r'output\S36_Ablation_Recognizer\AC.png')
    # print(img.shape)
    # cv2.imshow('cropped', img)
    # cv2.waitKey()
    # text = pytesseract.image_to_string(img)
    # print(text)

    #
    correctRecog = 0
    for iImg in range(test_images.shape[0]):
        img = np.squeeze(test_images[iImg, ...])
        # imgCrop = np.squeeze(test_images[iImg, 28:78, 26:78, ...])
        imgCrop = np.squeeze(test_images[iImg, 28:78, 24:80, ...])
        img = imgCrop

        label = test_labels[iImg, :]

        thresh1, ret = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        img = ret
        text = pytesseract.image_to_string(img)
        codeGD = inv_suit_dict[label[0]] + inv_suit_dict[label[1]]

        # cv2.imshow('cropped', img)
        # cv2.waitKey()

        if codeGD != text[:2]:
            print("Wrong detection, annotation: ", codeGD, " prediction: ", text)
            # cv2.imshow('cropped', img)
            # cv2.waitKey()
        else:
            correctRecog += 1
    print('Overall prediction accuracy: ', correctRecog / test_images.shape[0])

    # # Read image from which text needs to be extracted
    # # img = cv2.imread(r"output\S36_Ablation_Recognizer\sample4.jpg")
    # img = cv2.imread(r"output\S36_Ablation_Recognizer\sample4.jpg")
    #
    # # Preprocessing the image starts
    # # Convert the image to gray scale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # Performing OTSU threshold
    # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    #
    # # Specify structure shape and kernel size.
    # # Kernel size increases or decreases the area
    # # of the rectangle to be detected.
    # # A smaller value like (10, 10) will detect
    # # each word instead of a sentence.
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    #
    # # Appplying dilation on the threshold image
    # dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
    #
    # # Finding contours
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_NONE)
    #
    # # Creating a copy of image
    # im2 = img.copy()
    #
    # # A text file is created and flushed
    # file = open(join(outFolder, "recognized.txt"), "w+")
    # file.write("")
    # file.close()
    #
    # # Looping through the identified contours
    # # Then rectangular part is cropped and passed on
    # # to pytesseract for extracting text from it
    # # Extracted text is then written into the text file
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)
    #
    #     # Drawing a rectangle on copied image
    #     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     # Cropping the text block for giving input to OCRk
    #     cropped = im2[y:y + h, x:x + w]
    #
    #     cv2.imshow('cropped', cropped)
    #     cv2.waitKey()
    #     # Open the file in append mode
    #     file = open(join(outFolder, "recognized.txt"), "a")
    #
    #     # Apply OCR on the cropped image
    #     text = pytesseract.image_to_string(cropped)
    #
    #     # Appending the text into file
    #     file.write(text)
    #     file.write("\n")
    #
    #     # Close the file
    #     file.close