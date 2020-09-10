from Utility import *
import json
from pathlib import Path
import ComponentExtractor

class ForegroundSubtractorCleanPlate:
    def __init__(s, cleanPlate):
        s.cleanPlate=cleanPlate

    def __call__(s, img):
        return np.abs(img-s.cleanPlate)

def load_images(img_dir, UndistImgs=False, camParamF=None, cropSize=2160, imgExt='png', writeUndistorted=True,
                normalize=True, flipImg=True, cvtToRGB=True):
    image_refs_out = []
    crops_out = []
    undistImageFolder = join(img_dir, 'Undist')

    if UndistImgs:
        os.makedirs(undistImageFolder, exist_ok=True)
        camParams = json.load(open(camParamF))['cam_params']

    # for img_name in img_names:
    #    path = img_dir + '\\{}'.format(img_name)
    #    print(path)
    img_paths = sorted(glob.glob(img_dir + '/*.' + imgExt))

    for i, path in enumerate(img_paths):
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img = cv2.imread(path)

        if UndistImgs:
            # f = inFiles[iCam][iP]
            fx = camParams[str(i)]['fx']
            fy = camParams[str(i)]['fy']
            cx = camParams[str(i)]['cx']
            cy = camParams[str(i)]['cy']
            intrinsic_mtx = np.array([
                [fx, 0.0, cx, ],
                [0.0, fy, cy],
                [0.0, 0.0, 1],
            ])

            undistortParameter = np.array(
                [camParams[str(i)]['k1'], camParams[str(i)]['k2'], camParams[str(i)]['p1'], camParams[str(i)]['p2'],
                 camParams[str(i)]['k3'], camParams[str(i)]['k4'], camParams[str(i)]['k5'], camParams[str(i)]['k6']])

            img = cv2.undistort(img, intrinsic_mtx, undistortParameter)
            if writeUndistorted:
                outUndistImgFile = join(undistImageFolder, Path(path).stem + '.png')
                cv2.imwrite(outUndistImgFile, img)
        if normalize:
            img = img.astype(np.float32) / 255.0
        if cvtToRGB:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_refs_out.append(img)

    w = int(img.shape[0]) / 2

    for i in range(len(image_refs_out)):
        image = image_refs_out[i]
        cx = image.shape[1] / 2

        image = image_refs_out[i]
        img = image[:, int(cx - w):int(cx + w)]
        if not cropSize == img.shape[0]:
            img = cv2.resize(img, (cropSize, cropSize))
        if flipImg:
            img = cv2.flip(img, -1)
        crops_out.append(img)

    return image_refs_out, crops_out

def removeSmallBlackComponents( img, color = 255, whiteComponentsSizeLowerBound= 900):
    whiteComponents = ComponentExtractor.componentsExtractor(img, color, whiteComponentsSizeLowerBound)
    for component in whiteComponents:
        if len(component) < whiteComponentsSizeLowerBound:
            component = [(c[1], c[0]) for c in component]
            coords = tuple(np.array(component).T)
            # print("Remove components with size: ", len(component))
            img[coords] = 0

def foregroundSubtractionNaive(img, bg, diffThre):
    diffImg = np.abs((img.astype(np.float32) - bg.astype(np.float32)))

    # cv2.imshow('Diff', diffImg.astype(np.uint8))
    # cv2.waitKey()

    diffImgNorm = np.sqrt(diffImg[:, :, 0] ** 2 + diffImg[:, :, 1] ** 2 + diffImg[:, :, 2] ** 2)
    bgImgNorm = np.sqrt(imgCP[:, :, 0] ** 2 + imgCP[:, :, 1] ** 2 + imgCP[:, :, 2] ** 2)

    # diffRelative = 255 * diffImgNorm / (bgImgNorm + 1)
    # diffRelative[np.where(diffRelative > 255)] = 255

    # cv2.imshow('DiffNorm', diffImgNorm.astype(np.uint8))
    # cv2.imshow('diffRelative', diffRelative.astype(np.uint8))
    # cv2.waitKey()

    # diffImgNorm[np.where(diffRelative > threshold)] = 255
    diffMask = np.zeros((diffImgNorm.shape), dtype=np.float32)
    diffMask[np.where(diffImgNorm > diffThre)] = 255
    return diffMask

if __name__ == '__main__':
    import cv2

    inputRenderedImg = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\Rendered\08564.png'
    inputRenderedFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\Rendered'
    inputCleanPlateImg = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist\A.png'
    inputCleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    out_img_path_MOG = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRefMOG_Denoised'
    out_img_path_naive = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\ForeGroundRef_Naive_Denoised'

    cp_out, cp_crop_out = load_images(inputCleanPlateFolder, cropSize=1080, UndistImgs=False,
                                      camParamF=None, imgExt='png', flipImg=False, normalize=False)
    img = cv2.imread(inputRenderedImg)
    imgCP = cp_crop_out[0]
    naiveSubtraction = True
    MOGSubtraction = False
    diffThres = 15

    allImgFiles = sortedGlob(join(inputRenderedFolder, '*.png'))
    os.makedirs(out_img_path_naive, exist_ok=True)
    if naiveSubtraction:
        imgCP = cv2.bilateralFilter(imgCP, 15, 75, 75)

        for imgF in allImgFiles:
            img = cv2.imread(imgF)
            img = cv2.bilateralFilter(img, 15, 75, 75)

            # cv2.imshow('img', img)
            # cv2.imshow('imgCP', imgCP)
            # cv2.waitKey()

            diffImg = np.abs((img.astype(np.float32) - imgCP.astype(np.float32)))

            # cv2.imshow('Diff', diffImg.astype(np.uint8))
            # cv2.waitKey()

            diffImgNorm = np.sqrt(diffImg[:,:,0]**2 + diffImg[:,:,1]**2 + diffImg[:,:,2]**2)
            bgImgNorm = np.sqrt(imgCP[:,:,0]**2 + imgCP[:,:,1]**2 + imgCP[:,:,2]**2)

            foreground = foregroundSubtractionNaive(img, imgCP, diffThres)
            cv2.imshow('Foreground before noise removal', foreground.astype(np.uint8))

            # kernel = np.ones((3, 3), np.uint8)
            #
            # fgMask = cv2.dilate(foreground, kernel, iterations=2)
            # fgMaskInverse = 255 - fgMask
            # removeSmallBlackComponents(fgMaskInverse)
            # fgMaskDenoised = 255 - fgMaskInverse
            #
            # fgMaskDenoised = cv2.erode(fgMaskDenoised, kernel, iterations=2)
            # removeSmallBlackComponents(fgMaskDenoised, whiteComponentsSizeLowerBound=2000)

            fgMask = foreground
            removeSmallBlackComponents(fgMask, whiteComponentsSizeLowerBound=10)
            fgMaskInverse = 255 - fgMask
            removeSmallBlackComponents(fgMaskInverse, whiteComponentsSizeLowerBound=400)
            fgMaskDenoised = 255 - fgMaskInverse

            removeSmallBlackComponents(fgMaskDenoised, whiteComponentsSizeLowerBound=2000)


            cv2.imshow('FG Mask after component removal', fgMaskDenoised)
            cv2.waitKey(20)

            fName = os.path.basename(imgF)

            cv2.imwrite(join(out_img_path_naive, fName), (fgMaskDenoised ).astype(np.uint8))

        # cv2.imshow('Diff', diffImg.astype(np.uint8))
        # cv2.waitKey()
        # diffRelative = 255* diffImgNorm / (bgImgNorm+1)
        # diffRelative[np.where(diffRelative>255)] = 255
        #
        # cv2.imshow('DiffNorm', diffImgNorm.astype(np.uint8))
        # cv2.imshow('diffRelative', diffRelative.astype(np.uint8))
        # cv2.waitKey()
        #
        # # diffMask10 = np.zeros((diffImgNorm.shape), dtype=np.float32)
        # # diffMask10[np.where(diffImgNorm>10)] = 255
        # # cv2.imshow('diffMask10', diffMask10.astype(np.uint8))
        # #
        # # cv2.waitKey()
        #
        # diffThresAll = [ 10 * i for i in range(1,25)]
        #
        # for diffThre in diffThresAll:
        #     diffMask = np.zeros((diffImgNorm.shape), dtype=np.float32)
        #     diffMask[np.where(diffRelative>diffThre)] = 255
        #     cv2.imshow('diffMask' + str(diffThre), diffMask.astype(np.uint8))
        #
        #     cv2.waitKey()

    os.makedirs(out_img_path_MOG, exist_ok=True)
    if MOGSubtraction:

        img = cv2.bilateralFilter(img, 15, 75, 75)
        imgCP = cv2.bilateralFilter(imgCP, 15, 75, 75)

        backSub = cv2.createBackgroundSubtractorMOG2()
        # backSub = cv2.createBackgroundSubtractorKNN()
        fgMask = backSub.apply(imgCP)

        # cv2.imshow('FG Mask', fgMask)
        # cv2.waitKey()

        for imgF in allImgFiles:
            img = cv2.imread(imgF)
            fgMask = backSub.apply(img)
            fgMask[np.where(fgMask)] = 255

            cv2.imshow('FG Mask before component removal', fgMask)


            kernel = np.ones((3, 3), np.uint8)

            fgMask = cv2.dilate(fgMask, kernel, iterations=2)
            fgMaskInverse = 255 - fgMask
            removeSmallBlackComponents(fgMaskInverse)
            fgMaskDenoised = 255 - fgMaskInverse

            fgMaskDenoised = cv2.erode(fgMaskDenoised, kernel, iterations=2)
            removeSmallBlackComponents(fgMaskDenoised, whiteComponentsSizeLowerBound=900)

            cv2.imshow('FG Mask after component removal', fgMaskDenoised)

            cv2.waitKey()

            fName = os.path.basename(imgF)

            # outSilhouetteFile = join(out_img_path_MOG, fName)
            cv2.imwrite(join(out_img_path_MOG, fName), (fgMaskDenoised ).astype(np.uint8))
