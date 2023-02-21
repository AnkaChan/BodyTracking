from FlowNet2Wrapper import Flownet2Controller
# import FlowNet2Wrapper
import cv2, json
from Utility import *

def preprocessImg(img, camParam, turnToRGB=False):
    # convert to Rgb
    # Undist images
    # img = cv2.imread(inImgFile, cv2.IMREAD_GRAYSCALE)
    if turnToRGB:
        imgColor = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA )

    fx = camParam['fx']
    fy = camParam['fy']
    cx = camParam['cx']
    cy = camParam['cy']
    intrinsic_mtx = np.array([
        [fx, 0.0, cx, ],
        [0.0, fy, cy],
        [0.0, 0.0, 1],
    ])

    undistortParameter = np.array(
        [camParam['k1'], camParam['k2'], camParam['p1'], camParam['p2'],
         camParam['k3'], camParam['k4'], camParam['k5'], camParam['k6']])

    imgColorUndist = cv2.undistort(imgColor, intrinsic_mtx, undistortParameter)

    return imgColorUndist


def genOpticalFlow(inRenderedImgFolder, outFlowFolder, refImgFolder, frameNames=None, magAmplify=400, inRealImgExt='png'):
    inRenderedImgCamFolders = sortedGlob(join(inRenderedImgFolder, '*'))
    camNames = [os.path.basename(camFolder) for camFolder in inRenderedImgCamFolders]

    for camName, inRenderedImgCamFolder in zip(camNames, inRenderedImgCamFolders):
        refImgCamFolder = join(refImgFolder, camName, 'Reference')

        if frameNames is None:
            renderedImgFiles = sortedGlob(join(inRenderedImgCamFolder,'Rendered', '*.png'))
        else:
            renderedImgFiles = [join(inRenderedImgCamFolder, 'Rendered', frameName + '.png') for frameName in frameNames]

        outFlowCamFolder = join(outFlowFolder, 'Flow', camName, )
        outFlowImgCamFolder = join(outFlowFolder, 'Image', camName)

        os.makedirs(outFlowCamFolder, exist_ok=True)
        os.makedirs(outFlowImgCamFolder, exist_ok=True)

        for renderedImgF in tqdm.tqdm(renderedImgFiles, desc='Processing cam ' + camName):
            fileName = os.path.basename(renderedImgF)
            refImgFile = join(refImgCamFolder,  camName + fileName + '.png')
            im1 = cv2.imread(renderedImgF)
            im2 = cv2.imread(refImgFile)

            flow = flow_controller.predict(im1, im2)
            flow_image = flow_controller.convert_flow_to_image(flow, magAmplify=magAmplify)

            outFlowFile = join(outFlowCamFolder, fileName + '.npy')
            outFlowImgFile = join(outFlowImgCamFolder, fileName + '.png')

            np.save(outFlowFile, flow, )
            cv2.imwrite(outFlowImgFile, flow_image)



def genOpticalFlowForOneFrame(inRenderedImgFile, refImgFile, outFlowFile, outFlowImgFile, flow_controller, camParam, magAmplify=200,
                              undistorRefImg=True, convertRefImgToRGB=False, cleanPlate=None, outRefImgF=None, outSynthImgF=None, saveAlphaChannel=True):
    imSynth = cv2.imread(inRenderedImgFile, cv2.IMREAD_UNCHANGED)
    imRef = cv2.imread(refImgFile, cv2.IMREAD_UNCHANGED)

    if cleanPlate is not None:
        alphaBackgroundMask = np.where(imSynth[:,:,3] == 0)

        imSynth[alphaBackgroundMask[0],alphaBackgroundMask[1], :3] = cleanPlate[alphaBackgroundMask[0],alphaBackgroundMask[1], :]

        # cv2.imshow('SynthImg', imSynth)
        # cv2.waitKey()

    if undistorRefImg:
        imRef = preprocessImg(imRef, camParam, convertRefImgToRGB)

    imRef = cv2.resize(imRef, (imSynth.shape[1], imSynth.shape[0]) )

    if outRefImgF is not None:
        cv2.imwrite(outRefImgF, imRef)

    if outSynthImgF is not None:
        if saveAlphaChannel:
            cv2.imwrite(outSynthImgF, imSynth)
        else:
            cv2.imwrite(outSynthImgF, imSynth[:,:,:3])

    flow = flow_controller.predict(imSynth[:,:, :3], imRef)
    flow_image = flow_controller.convert_flow_to_image(flow, magAmplify=magAmplify)

    np.save(outFlowFile, flow, )
    if outFlowImgFile is not None:
        cv2.imwrite(outFlowImgFile, flow_image)



if __name__ == '__main__':
    inCamParam = r'/media/Data001/MocapProj/2019_12_13_Lada_Capture/CameraParameters/cam_params.json'
    inImageFolder = r'/media/Data001/MocapProj/2019_12_13_Lada_Capture'
    # inSynthImgFolder = r'/media/Data001/MocapProj/2021_01_16_OpticalFlows/Lada_Ground'
    inCleanPlateFolder = r'/media/Data001/MocapProj/2019_12_13_Lada_Capture/CleanPlateExtracted/gray/distorted/RgbUndist'
    # inSynthImgFolder = r'/media/anka/TOSHIBA EXT/2021_01_17_SyntheticImages_LadaGround/LadaGround'
    # outFolder = r'/media/Data001/MocapProj/2021_01_16_OpticalFlows2/Lada_Ground'

    inSynthImgFolder = r'/media/anka/Chenhe/2021_01_17_SyntheticImages_LadaGround_LBS'
    outFolder = r'/media/Data001/MocapProj/2021_01_16_OpticalFlows2/Lada_Ground_LBS'

    flowNetCkpFile = r'/home/anka/Code/00_DeepLearning/flownet2-pytorch/FlowNet2_checkpoint.pth.tar'

    camIds = [0,4,8,12]
    camNames = 'ABCDEFGHIJKLMNOP'
    # frameNames = ['08102', '08034']
    frameNames = [str(frame).zfill(5) for frame in range(6141, 8141)]

    # outputRefImg = True
    # outputOverlayedSynthetic = True
    # outFlowImg = True

    outputRefImg = False
    outputOverlayedSynthetic = False
    outFlowImg = False

    flow_controller = Flownet2Controller.FlowController(flowNetCkpFile)

    camParamsAll = json.load(open(inCamParam))

    if outputRefImg:
        refImgFolder = join(outFolder, 'Ref')
        os.makedirs(refImgFolder, exist_ok=True)
    if outputOverlayedSynthetic:
        synthImgFolder = join(outFolder, 'Synth')
        os.makedirs(synthImgFolder, exist_ok=True)

    flowImgFolder = join(outFolder, "FlowImage")
    flowFolder = join(outFolder, "Flow")

    os.makedirs(flowImgFolder, exist_ok=True)
    os.makedirs(flowFolder, exist_ok=True)

    for iCam in camIds:
        camName = camNames[iCam]
        inCleanPlateImg = cv2.imread(join(inCleanPlateFolder, camName + '.png'))

        camParam = camParamsAll['cam_params'][str(iCam)]

        inCleanPlateImg = cv2.resize(inCleanPlateImg, (2000, 1080,))

        for frame in  tqdm.tqdm(frameNames, desc='Generating for cam: '+camName):
            inRefImgF = join(inImageFolder, camName, camName + frame + '.pgm')
            inSynthImgF = join(inSynthImgFolder, camName + frame + '.png')

            outputRefImgF = join(refImgFolder, camName + frame + '.png') if outputRefImg else None
            outputSynthImgF = join(synthImgFolder, camName + frame + '.png') if outputRefImg else None

            outFlowFile = join(flowFolder, camName + frame + '.npy')
            outFlowImgFile = join(flowImgFolder, camName + frame + '.png') if outFlowImg else None

            genOpticalFlowForOneFrame(inSynthImgF, inRefImgF, outFlowFile, outFlowImgFile, flow_controller, camParam,
                                      cleanPlate=inCleanPlateImg, undistorRefImg=True, convertRefImgToRGB=True,
                                      outRefImgF=outputRefImgF, outSynthImgF=outputSynthImgF)


    # frameNames = [
    #     str(i).zfill(5) for i in range(8564, 8564 + 200)]

