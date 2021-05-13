from S33_Evaluate_EndToEnd_Test import *

if __name__ == '__main__':
    inFolder = r'D:\GDrive\mocap\2019_06_05_NewSuitCapture2\Converted\D'
    processName = r'Pattern_quad_proposal_3660_5200'
    outFolder = join(r'output', 'S37_StudyReconForNewSuit')

    frameNames = list(range(3660, 5200, 100))

    frameNames = [str(fId).zfill(5) for fId in frameNames]

    for frameName in frameNames:
        imgFile = join(inFolder, frameName + '.pgm')
        predFile = join(inFolder, processName, frameName + '.json')
        outPDFName = join(outFolder, frameName + '.pdf')
        img = cv2.imread(imgFile)
        drawRecogResultsPDF(outPDFName, img, predFile, drawCorners=True)



