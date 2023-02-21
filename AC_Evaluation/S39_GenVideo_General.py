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
    # inSynthImgFolder = r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories\Jump'
    # outFolderFile = join(r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories', 'Jump.mp4')

    # inSynthImgFolder = r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories\BackBend'
    # outFolderFile = join(r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories', 'BackBend.mp4')

    inSynthImgFolder = r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories\SitSpin'
    outFolderFile = join(r'F:\WorkingCopy2\2021_05_12_ReverseRenderingAnimation\Trajectories', 'SitSpin.mp4')

    fps = 30


    outputSize = (1920, 1080,)

    out = cv2.VideoWriter(outFolderFile, cv2.VideoWriter_fourcc(*'mp4v'), fps, outputSize)
    # out = cv2.VideoWriter(join(outFolder, 'synth.avi'), cv2.VideoWriter_fourcc(*'MJPG'), fps, outputSize)
    frameNames = sorted(glob.glob(join(inSynthImgFolder, '*.png')))

    for frame in  tqdm.tqdm(frameNames, ):
        # inSynthImgF = join(inSynthImgFolder, camName + frame + '.png')
        imSynth = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
        imSynth = cv2.resize(imSynth, (int(outputSize[0]), int(outputSize[1])))

        # alphaBackgroundMask = np.where(imSynth[:, :, 3] == 0)
        # imSynth[alphaBackgroundMask[0], alphaBackgroundMask[1], :3] = inCleanPlateImg[alphaBackgroundMask[0],
        #                                                               alphaBackgroundMask[1], :]
        cv2.imshow('imSynth', imSynth,)
        cv2.waitKey(20)
        out.write(imSynth)
    out.release()
