import cv2, glob
import tqdm
import glob, os, json
from os.path import join
import numpy as np
from pathlib import Path
def sortedGlob(pathname):
    return sorted(glob.glob(pathname))


if __name__ == '__main__':
    # inputRealImgFolder = r'F:\WorkingCopy2\2021_01_12_AC_FilesForWireframes\Copied_Katey\Images\09246'
    # inputRenderedFiles = r'Rendered\Katey\ManuallyEdited'
    # outFolder = r'Overlay/Katey'
    # camFile = r'F:\WorkingCopy2\2020_01_01_KateyCapture\CameraParameters\cam_params.json'
    inputRealImgFolderAll = r'F:\WorkingCopy2\2021_01_12_AC_FilesForWireframes\Copied_Lada\Images'
    inputRenderedFilesAll = r'E:\WorkingCopy\2021_01_13_TemporalSmoothingTuning\Rendering'
    outFolder =  r'E:\WorkingCopy\2021_01_13_TemporalSmoothingTuning\Overlay'
    seqName = 'Old'
    camSelected = 'G'
    iCam = 6
    camFile = r'F:\WorkingCopy2\2019_12_13_Lada_Capture\CameraParameters\cam_params.json'

    resize= (2000, 1080)

    # inputFrameNames =["11071", "12847", "13550"]

    # inputRealImgFolderAll = r'F:\WorkingCopy2\2021_01_12_AC_FilesForWireframes\Copied_Marianne\Images'
    # inputRenderedFilesAll = r'Rendered\Marianne'
    # outFolder = r'Overlay/Marianne'
    # camFile = r'F:\WorkingCopy2\2019_12_24_Marianne_Capture\CameraParameters\cam_params.json'

    inputFrameNames = [str(i).zfill(5) for i in range(9224, 9335)]
    inputRenderedFolder = join(inputRenderedFilesAll, seqName)
    outFolder = join(outFolder, seqName)
    os.makedirs(outFolder, exist_ok=True)
    camParamAll = json.load(open(camFile))

    for frameName in tqdm.tqdm(inputFrameNames):
        inputRealImgFolder = join(inputRealImgFolderAll, frameName)

        imgF = join(inputRealImgFolder, camSelected + frameName + '.pgm')
        renderedImgF = join(inputRenderedFolder, camSelected + frameName + '.png')

        # for iCam, (imgF, renderedImgF) in enumerate(zip(imgFiles, renderImgFiles)):
        img = cv2.imread(imgF, cv2.IMREAD_COLOR)
        renderedImg = cv2.imread(renderedImgF, cv2.IMREAD_COLOR)

        camParam = camParamAll['cam_params'][str(iCam)]
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

        img = cv2.undistort(img, intrinsic_mtx, undistortParameter)
        # img = cv2.resize(img, resize)
        # renderMask = np.logical_and(np.logical_and(renderedImg[:,:,0], renderedImg[:,:,1]), renderedImg[:,:,2])
        renderMask = renderedImg[:,:,1] > 100
        # for iChannel in range(3):
        for iChannel in [1]: # only copy the color of Green
            imageChannel = img[:,:, iChannel]
            renderMaskChannel = renderedImg[:,:, iChannel]
            imageChannel[np.where(renderMask)] = renderMaskChannel[renderMask]
            img[:, :, iChannel] = imageChannel
            # img[:,:, iChannel] = renderedImg[:,:, iChannel]

        # cv2.imshow('Overlayed', img)
        # cv2.waitKey(0)

        cv2.imwrite(join(outFolder, os.path.basename(imgF) + '.png'), img)