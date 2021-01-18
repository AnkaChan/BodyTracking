from FlowNet2Wrapper import Flownet2Controller
# import FlowNet2Wrapper
import cv2
from Utility import *

def genOpticalFlow(inRenderedImgFolder, outFlowFolder, refImgFolder, frameNames=None, magAmplify=400):
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


if __name__ == '__main__':

    flowNetCkpFile = r'/home/anka/Code/00_DeepLearning/flownet2-pytorch/FlowNet2_checkpoint.pth.tar'

    flow_controller = Flownet2Controller.FlowController(flowNetCkpFile)

    # im1 = cv2.imread(
    #     r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/HandOnGround/A/FlipComparison/A08564.0Rendered.png')
    # im1 = cv2.imread(r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/PureLBS/Rendered/A/Rendered/08564.png')

    # im1 = cv2.imread(r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/Rendered/A/Rendered/08564.png')
    # im2 = cv2.imread(
    #     r"/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/HandOnGround/A/FlipComparison/A08564.png.png")
    #
    # # for i in tqdm.tqdm(range(100)):
    # #     flow = flow_controller.predict(im1, im2)
    # flow = flow_controller.predict(im1, im2)
    # # Important note : All predictions are made at maximum viable resolution to ensure prediction quality is high,
    # # but this comes at a massive hit to performance, if you want fast executions I suggest downsampling images first
    #
    # # Can convert flow to image using built in method
    # flow_image = flow_controller.convert_flow_to_image(flow)
    #
    # cv2.imshow("Random flow image", flow_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #
    # # Can also convert video's to their optical flow variants using following method
    # # Use raw=true argument for only saving the optical flow video
    # # Set downsample_res if you want to process video faster
    # flow_controller.convert_video_to_flow("cp77cinematic.mp4", "output", downsample_res=(320, 320))

    frameNames = [
        str(i).zfill(5) for i in range(8564, 8564 + 200)]

    inLBSMeshRenderingFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/ToTrackingPoints/Rendered'
    outLBSMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/ToTrackingPoints/OpticalFlow'
    referenceImageFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered'
    genOpticalFlow(inLBSMeshRenderingFolder, outLBSMeshOpticalFlowFolder, referenceImageFolder, frameNames)

    inLBSMeshRenderingFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/ToDense/Rendered'
    outLBSMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/ToDense/OpticalFlow'
    referenceImageFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered'
    genOpticalFlow(inLBSMeshRenderingFolder, outLBSMeshOpticalFlowFolder, referenceImageFolder, frameNames)

    # inInterpolatedMeshRenderingFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/Rendered'
    # outInterpolateMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/Evaluation/Interpolated/OpticalFlow'
    # genOpticalFlow(inInterpolatedMeshRenderingFolder, outInterpolateMeshOpticalFlowFolder, referenceImageFolder, frameNames)

    # inFinalMeshRenderingFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/'
    # outFinalMeshOpticalFlowFolder = r'/media/Data001/MocapProj/2020_08_24_TexturedFitting_Lada_Rendering/DiffRenderered/OpticalFlow'
    # genOpticalFlow(inFinalMeshRenderingFolder, outFinalMeshOpticalFlowFolder, referenceImageFolder, frameNames)
