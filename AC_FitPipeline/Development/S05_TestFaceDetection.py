from Utility_Development import *

from M02_ReconstructionJointFromRealImagesMultiFolder import *

if __name__ == '__main__':
    inFolder = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TPose\Preprocessed\18331'
    openposeModelDir = r"C:\Code\Project\Openpose\models"
    outFolder = join(inFolder, 'FaceKps')

    os.makedirs(outFolder, exist_ok=True)

    params = dict()
    params["model_folder"] = openposeModelDir
    params["face"] =True
    params["hand"] = True

    convertToRGB = False

    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    imgFiles = sortedGlob(join(inFolder, '*.png'))
    # imgFiles = sortedGlob(join(inFolder, '*.jpg'))

    if convertToRGB:
        examplePngFiles = glob.glob(join('ExampleFiles', '*.dng'))
        imgs = inverseConvertMultiCams(imgFiles, None, examplePngFiles, writeFiles=False)
    else:
        imgs = [cv2.imread(imgF) for imgF in imgFiles]

    for img, imgF in zip(imgs, imgFiles):
        # imageToProcess = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imageToProcess = img
        imageToProcess = cv2.resize(imageToProcess, (imageToProcess.shape[1]//2, imageToProcess.shape[0]//2))

        datum = op.Datum()
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])
        data = {}

        data['BodyKeypoints'] = datum.poseKeypoints.tolist()

        fig, ax = plt.subplots()
        imageToProcess = cv2.cvtColor(imageToProcess, cv2.COLOR_BGR2RGB)
        ax.imshow(imageToProcess, vmin=0, vmax=255, interpolation='nearest',)

        for iF in range(datum.faceKeypoints.shape[0]):
            faceKp = datum.faceKeypoints[0,:,:]
            print(faceKp.shape)
            faceKp = faceKp[np.where(faceKp[:,2]>0)[0], :2]

            ax.plot(faceKp[:, 0], faceKp[:, 1], 'x', color='green', markeredgewidth=0.15, markersize=0.8)  # markeredgewidth=0.06, markersize=0.3
            ax.axis('off')

            imgName = os.path.basename(imgF) + '.' + str(iF) + '.png'

            outImgFile = join(outFolder, imgName)
            fig.savefig(outImgFile, dpi=1000, bbox_inches='tight', pad_inches=0)


