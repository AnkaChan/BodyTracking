from S29_3DReconstructedPointsOpticalFlow import *
from matplotlib import pyplot as plt

def computeOpticalFlowErrorAllPixels(flow, sil):
    foregroundPixels = np.where(sil)

    # flowImg = convert_flow_to_image(flow, )
    # cv2.imshow('Sil', sil)
    # cv2.imshow('flowImg', flowImg)
    # cv2.waitKey()

    flowMag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    # avgFlowNorm = np.mean(flowMag[foregroundPixels])

    return flowMag[foregroundPixels]

def readOpticalFlows(flowFiles, masks=None):
    flows = []
    if masks is None:
        for flowFile in flowFiles:
            flow = np.load(flowFile)
            flows.append(flow)
    else:
        for flowFile, mask in tqdm.tqdm(zip(flowFiles, masks)):
            flow = np.load(flowFile)
            flows.append(flow)

            # mask = cv2.imread(maskFile, cv2.IMREAD_GRAYSCALE)
            flow[np.where(np.logical_not(mask))] = 0
            masks.append(mask)
            # cv2.imshow('MaskedFlow', np.abs(flow[:,:,0])*10, )
            # cv2.waitKey()

    return flows

def getLocalizationErrors(errs):

    dis = errs

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
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def convert_flow_to_image(flow, magAmplify=40):
    image_shape = flow.shape[0:2] + (3,)

    hsv = np.zeros(shape=image_shape, dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    normalized_mag = np.asarray(np.clip(mag*magAmplify, 0, 255), dtype=np.uint8)
    hsv[..., 2] = normalized_mag
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    rgb = np.asarray(rgb, np.uint8)
    return rgb

if __name__ == '__main__':
    inCorrsFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\CorrsType1Only'
    inTriangulationFolder = r'F:\WorkingCopy2\2020_03_18_LadaAnimationWholeSeq\WholeSeq\TriangulationType1Only'
    calibrationDataFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CameraParameters\cam_params.json'
    flowFolder = r'X:\MocapProj\2021_01_16_OpticalFlows2\Lada_Ground\Flow'
    synthImgsFolder = r'G:\2021_01_17_SyntheticImages_LadaGround\LadaGround'
    outputFolder = r'E:\WorkingCopy\2021_01_18_OpticalFlowAnalysis\LadaGround'

    resizeLvl = 0.5
    # frames = [str(iFrame).zfill(5) for iFrame in range(6141, 6141+2000)]
    # frames = [str(iFrame).zfill(5) for iFrame in range(7032, 6141+2000)]
    # frames = [str(iFrame).zfill(5) for iFrame in range(7032, 6141+2000)]

    frames = [str(iFrame).zfill(5) for iFrame in range(6141, 6141+2000, 10)]


    # what to report
    # 1. average optical flow norms on each frame each camera
    # 2. optical flow errors for each observed points on each camera that sees it
    # 3.

    os.makedirs(outputFolder, exist_ok=True)

    camIds = [0, 4, 8, 12]
    camNames = 'ABCDEFGHIJKLMNOP'


    for iF, frame in tqdm.tqdm(enumerate(frames)):
        corrFile = join(inCorrsFolder, 'A' + frame.zfill(8) + '.json')
        corrs = json.load(open(corrFile))

        camIdsObserved = corrs['camIdsUsed']

        triangulationFile = join(inTriangulationFolder, 'A' + frame.zfill(8) + '.obj')
        triangulation = pv.PolyData(triangulationFile)

        avgOpticalFlowNorms = []  # nFrames x nCams

        for camId in camIds:
            cName = camNames[camId]

            flow = np.load(join(flowFolder, cName + frame + '.npy'))
            silFile = join(synthImgsFolder, cName + frame + '.png')
            sil = cv2.imread(silFile, cv2.IMREAD_UNCHANGED)

            # flowImg = convert_flow_to_image(flow, )
            # fig, ax = plt.subplots()
            # ax.imshow(sil[:,:,3], vmin=0, vmax=255, interpolation='nearest')

            allErrs = computeOpticalFlowErrorAllPixels(flow, sil[:,:,3])
            avgOpticalFlowNorms.append(allErrs)

            # for iV in range(pts2D.shape[0]):
            #     if camId in camIdsObserved[iV]:
            #         # print(flow[int(pts2D[iV, 1] * resizeLvl), int(pts2D[iV, 0] * resizeLvl), :])
            #         flowErr = bilinear_interpolate(flow, pts2D[iV, 0] * resizeLvl, pts2D[iV, 1] * resizeLvl)
            #         # print(flowErr)
            #         opticalFlowErrs.append(flowErr)
            #     else:
            #         opticalFlowErrs.append([0,0])

    # print(opticalFlowErrs)
    # print(avgOpticalFlowNorms)
        np.save(join(outputFolder, 'opticalFlowErrsAllPixels'+'_' + frame +'.npy'), np.array(allErrs))
        # np.save(join(outputFolder, 'avgOpticalFlowNorms'+'_' + frame +'.npy'), np.array(avgOpticalFlowNorms))

