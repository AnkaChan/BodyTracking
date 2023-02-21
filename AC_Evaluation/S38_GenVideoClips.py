# import FlowNet2Wrapper
import cv2, json
from Utility import *
from os.path import isfile, join

def imgs2Vid(pathIn, vidOut, fps = 30, imgScale = 1, select = []):

    #pathIn = r'F:\WorkingCopy2\2019_04_16_8CamsCapture\VideoSequence\D\\'
    #pathOut = r'F:\WorkingCopy2\2019_04_16_8CamsCapture\VideoSequence\D.avi'
    #fps = 30

    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])
    files.sort()
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    # for sorting the file names properly
    files.sort(key=lambda x: x[5:-4])

    if len(select) != 2:
        iterRange = range(len(files))
    else:
        iterRange = range(select[0], select[1])

    for i in iterRange:
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        newX, newY = img.shape[1] * imgScale, img.shape[0] * imgScale
        newimg = cv2.resize(img, (int(newX), int(newY)))
        height, width, layers = newimg.shape
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(newimg)
        #fourcc = cv2.VideoWriter_fourcc(*'')
    out = cv2.VideoWriter(vidOut, cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    inSynthImgFolder = r'H:\2021_05_12_SyntheticImages_LadaGround_Stable'
    inCleanPlateFolder = r'X:/MocapProj/2019_12_13_Lada_Capture/CleanPlateExtracted/gray/distorted/RgbUndist'
    inCamParam = r'X:/MocapProj/2019_12_13_Lada_Capture/CameraParameters/cam_params.json'

    outFolder = r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Animation'
    outputSize = (2000, 1080,)
    fps = 30

    camIds = [8,]
    camNames = 'ABCDEFGHIJKLMNOP'
    # frameNames = ['08102', '08034']
    frameNames = [str(frame).zfill(5) for frame in range(6141, 8141)]
    # frameNames = [str(frame).zfill(5) for frame in range(6141, 6141+200)]

    # outputRefImg = True
    # outputOverlayedSynthetic = True
    # outFlowImg = True

    outputRefImg = False
    outputOverlayedSynthetic = False
    outFlowImg = False

    # flow_controller = Flownet2Controller.FlowController(flowNetCkpFile)
    # camParamsAll = json.load(open(inCamParam))

    os.makedirs(outFolder, exist_ok=True)

    out = cv2.VideoWriter(join(outFolder, 'synth.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, outputSize)
    # out = cv2.VideoWriter(join(outFolder, 'synth.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, outputSize)

    for iCam in camIds:
        camName = camNames[iCam]
        inCleanPlateImg = cv2.imread(join(inCleanPlateFolder, camName + '.png'))
        inCleanPlateImg = cv2.resize(inCleanPlateImg, outputSize)

        for frame in  tqdm.tqdm(frameNames, desc='Generating for cam: '+camName):

            inSynthImgF = join(inSynthImgFolder, camName + frame + '.png')
            imSynth = cv2.imread(inSynthImgF, cv2.IMREAD_UNCHANGED)
            alphaBackgroundMask = np.where(imSynth[:, :, 3] == 0)

            imSynth[alphaBackgroundMask[0], alphaBackgroundMask[1], :3] = inCleanPlateImg[alphaBackgroundMask[0],
                                                                          alphaBackgroundMask[1], :]
            # cv2.imshow('imSynth', imSynth,)
            # cv2.waitKey(20)
            out.write(imSynth[..., :3])
    out.release()
