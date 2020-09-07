from Utility import *
import cv2

def imagesToVideo(imgFiles, outVidFile):
    img =  cv2.imread(imgFiles[0])
    vid_out = cv2.VideoWriter(outVidFile, cv2.VideoWriter_fourcc(*"MP4V"), 30, (img.shape[1], img.shape[0]), isColor=True)
    for imgF in imgFiles:
        img = cv2.imread(imgF)
        vid_out.write(img)

    vid_out.release()

if __name__ == '__main__':
    inputRefFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\Reference'
    allImgFiles = sortedGlob(join(inputRefFolder, '*.png'))

    imagesToVideo(allImgFiles, r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\DiffRenderered\A\RefVideo.mp4')
