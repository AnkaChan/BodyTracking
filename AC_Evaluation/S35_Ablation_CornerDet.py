import cv2
import numpy as np
from Utility import *
import json
# from SuitCapture import Debug
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

def get_vertices(obj):
    """extracts vertices (points) from a labelme-json file"""
    vlist = []
    for i in range(len(obj['shapes'])):
        if obj['shapes'][i]['label'] != 'bound_poly' and len(obj['shapes'][i]['points'])==6:
            vlist.append(obj['shapes'][i]['points'][0])
    return np.array(vlist)

def cornerDetectorST(img, qualityLevel, minDistance, blockSize, useHarrisDetector):
    # Apply corner detection
    corners = cv2.goodFeaturesToTrack(img, 0, qualityLevel, minDistance, None, blockSize=blockSize,
                                      useHarrisDetector=useHarrisDetector)

    # refine corners in subpixel accuracy

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cornersRefined = cv2.cornerSubPix(img, np.float32(corners), refineWindowSize, (-1, -1), criteria)
    cornersRefined = np.squeeze(cornersRefined)

    return cornersRefined

def cornerDetectorHarris(img, ):
    # Apply corner detection
    dst = cv2.cornerHarris(img,2,3,0.04)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    # find centroids
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return corners

def cornersDetectorSuperPoint(fileName):
    processedFolder = r'X:\MocapProj\2021_04_07_CornerLabellingTestSet1'

    processedFile = join(processedFolder, fileName + '_corners_SuperPoint.json')

    corners = json.load(open(processedFile))

    return np.array(corners)

def cornerDetCNN(fileName):
    processedFolder = r'C:\Code\MyRepo\03_capture\BodyTracking\AC_Evaluation\output\S33_Evaluate_EndToEnd_Test'
    processedFile = join(processedFolder, fileName + 'predictionFile.json')

    predData = json.load(open(processedFile))

    return np.array(predData['corners'])

def drawCorners(img, corners, corners_PDF_file):
    fig, ax = plt.subplots()
    ax.imshow(img, vmin=0, vmax=255, interpolation='nearest', cmap='gray')
    ax.axis('off')
    # ax.plot(corners[:, 0], corners[:, 1], 'x', color='green', markeredgewidth=0.1, markersize=0.5)
    ax.plot(corners[:, 0], corners[:, 1], 'x', color='red', markeredgewidth=0.1, markersize=0.5)
    fig.savefig(corners_PDF_file, dpi=2000, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.close()


def analyseCornerDetection(cornersAnnot, cornersDetected):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cornersDetected)
    distances, indices = nbrs.kneighbors(cornersAnnot)

    matchIds = np.where(distances < cornerDetMatchThreshold)[0]

    # print('Actual Positive: ', matchIds.shape[0])
    # print('False Negative: ', cornersAnnot.shape[0] - matchIds.shape[0])
    # print('False Positive: ', cornersDetected.shape[0] - matchIds.shape[0])
    #
    # print('Mean localization error: ', np.mean(distances[matchIds]))

    return {
        'TruePositive': matchIds.shape[0],
        'FalseNegative': cornersAnnot.shape[0] - matchIds.shape[0],
        'FalsePositive': cornersDetected.shape[0] - matchIds.shape[0],
        'MeanLocalizationError': np.mean(distances[matchIds])
    }


if __name__ == '__main__':
    inFolder = r'Data\S35_Ablation_CornerDet'
    outFolder = r'output\S35_Ablation_CornerDet'
    # inCornerAnnotFiles = sortedGlob(join(inFolder, '*.json'))
    # inImgs = sortedGlob(join(inFolder, '*.pgm'))

    fnames_test = ["01758A"]

    # os.path.basename(__file__)

    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.10
    minDistance = 5
    blockSize = 7

    cornerDetMatchThreshold = 1.5

    useHarrisDetector = True
    # k = 0.04
    refineWindowSize = (5,5)

    os.makedirs(outFolder, exist_ok=True)

    for frameName in fnames_test:
        inCornerLabelFile = join(inFolder, frameName + '_corners.json')
        inImgFile = join(inFolder, frameName + '.pgm')
        cornersAnnot = get_vertices(json.load(open(inCornerLabelFile)))

        img = cv2.imread(inImgFile, cv2.IMREAD_GRAYSCALE)

        cornersST = cornerDetectorST(img, qualityLevel, minDistance, blockSize, useHarrisDetector)
        cornersHarris = cornerDetectorHarris(img)
        cornersSuperPoint = cornersDetectorSuperPoint(frameName)
        cornersCornerDet = cornerDetCNN(frameName)

        corners_Annot_PDF_file = join(outFolder, frameName + '_Annot.pdf')
        corners_CornerDet_PDF_file = join(outFolder, frameName + '_CornerDet.pdf')
        corners_ST_PDF_file = join(outFolder, frameName + '_ST.pdf')
        corners_Harris_PDF_file = join(outFolder, frameName + '_Harris.pdf')
        corners_SP_PDF_file = join(outFolder, frameName + '_SuperPoint.pdf')

        # drawCorners(img, cornersAnnot, corners_Annot_PDF_file)
        drawCorners(img, cornersCornerDet, corners_CornerDet_PDF_file)
        drawCorners(img, cornersST, corners_ST_PDF_file)
        drawCorners(img, cornersHarris, corners_Harris_PDF_file)
        drawCorners(img, cornersSuperPoint, corners_SP_PDF_file)

        print(cornersAnnot.shape[0], ' corners annotated.')

        print(cornersCornerDet.shape[0], ' corners detected by CornerDet. Statistics:')
        print(analyseCornerDetection(cornersAnnot, cornersCornerDet))

        print(cornersST.shape[0], ' corners detected by ST. Statistics:')
        print(analyseCornerDetection(cornersAnnot, cornersST))

        print(cornersHarris.shape[0], ' corners detected by Harris. Statistics:')
        print(analyseCornerDetection(cornersAnnot, cornersHarris))

        print(cornersSuperPoint.shape[0], ' corners detected by SuperPoint. Statistics:')
        print(analyseCornerDetection(cornersAnnot, cornersSuperPoint))

        # confusion matrix




