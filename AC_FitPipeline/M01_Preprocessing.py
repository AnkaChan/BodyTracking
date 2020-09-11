import numpy as np
import cv2, imageio
import glob
import os, json
import time
from multiprocessing import Pool
from os.path import join
from pathlib import Path
from Utility import *

class Configs:
    # black cut-off level, can be read from RAW files with dcraw or exiftool
    black_level = 355
    # white cut-off (saturation) level, can be read from RAW files with dcraw or exiftool
    white_level = 27500
    # image dimensions in the RAW files (including margins)
    height = 2192
    width = 4032
    # sensor margins to be removed from final images:
    margin_crop = 16
    # white balance multipliers for green, red, and blue Bayer filters:
    wb_scales = (1.0, 2.3, 1.5)
    # coefficient to multiply all intensity values to increase brightness:
    brightness_mult = 7.5

    # folder that contains sub-directories with DNG images; example folder name is "A001_01011738_C008"
    # src_folder = r"d:\mocap\2019_01_10_FourCamera"
    # src_folder = "D:/Mocap/01_25_2019FourCam"
    # src_folder = r"E:\Mocap\WorkingCopy\2019_02_22_5CamReconstruction"
    # src_fold er = r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture"
    src_folders = []
    startId = 0

    # folder that will contain the output sub-directories with converted images:
    # dst_folder = r"E:\Mocap\WorkingCopy\2019_02_22_5CamReconstruction\Transformed"
    dst_folder = ""
    # either -1 to process all files, or a cap on total number of images to process in each folder (good for testing):
    max_frames = -1

    # how many worker processes to launch in parallel
    num_processes = 6

def inverseConverGrayImg(grayImg, config=Configs()):

    M1 = np.array([[1, 0], [0, 0]])
    M2 = np.array([[0, 1], [0, 0]])
    M3 = np.array([[0, 0], [1, 0]])
    M4 = np.array([[0, 0], [0, 1]])
    filter1 = np.tile(M1, (int(config.height / 2), int(config.width / 2)))
    filter2 = np.tile(M2, (int(config.height / 2), int(config.width / 2)))
    filter3 = np.tile(M3, (int(config.height / 2), int(config.width / 2)))
    filter4 = np.tile(M4, (int(config.height / 2), int(config.width / 2)))

    config.wb_filter_scaled = (filter1 * config.wb_scales[0] + filter2 * config.wb_scales[1] +
                               filter3 * config.wb_scales[2] + filter4 * config.wb_scales[0])

    inverted = (img / 255) / config.wb_filter_scaled[config.margin_crop:(config.height - config.margin_crop),
                             config.margin_crop:(config.width - config.margin_crop)] + config.black_level

    inverted = inverted.astype(np.uint8)
    # inverted = cv2.copyMakeBorder(inverted, config.margin_crop, config.margin_crop, config.margin_crop,
    #                               config.margin_crop, cv2.BORDER_REFLECT)
    # raw_image = rimg.raw_image
    # raw_image[:] = inverted
    # rgbFile = join(outFolder, os.path.basename(imgF) + '.png')
    # rgb = rimg.postprocess()
    # rgb = rgb[config.margin_crop:(config.height - config.margin_crop),
    #       config.margin_crop:(config.width - config.margin_crop)]
    # imageio.imsave(rgbFile, rgb)

    return inverted

def preprocessImg(inImgFile, outUndistRgbImgFile, camParam, outDistRgbFile=None):
    # convert to Rgb
    # Undist images
    img = cv2.imread(inImgFile, cv2.IMREAD_GRAYSCALE)
    imgColor = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR_EA )

    if outDistRgbFile is not None:
        cv2.imwrite(outDistRgbFile, imgColor)

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

    cv2.imwrite(outUndistRgbImgFile, imgColorUndist)

def inverseConvertMultiCams(imgFilesToConvert, outFolder, exampleDngFiles, config=Configs(), writeFiles=True):
    import rawpy as raw
    if writeFiles:
        os.makedirs(outFolder, exist_ok=True)
    # imgFiles = glob.glob(join(imgFilesToConvert, '*.' + inExtname))
    assert len(imgFilesToConvert) == len(exampleDngFiles)

    M1 = np.array([[1, 0], [0, 0]])
    M2 = np.array([[0, 1], [0, 0]])
    M3 = np.array([[0, 0], [1, 0]])
    M4 = np.array([[0, 0], [0, 1]])
    filter1 = np.tile(M1, (int(config.height / 2), int(config.width / 2)))
    filter2 = np.tile(M2, (int(config.height / 2), int(config.width / 2)))
    filter3 = np.tile(M3, (int(config.height / 2), int(config.width / 2)))
    filter4 = np.tile(M4, (int(config.height / 2), int(config.width / 2)))

    config.wb_filter_scaled = config.brightness_mult * \
                              (filter1 * config.wb_scales[0] + filter2 * config.wb_scales[1] +
                               filter3 * config.wb_scales[2] + filter4 * config.wb_scales[0]) \
                              / (config.white_level - config.black_level)

    rgbImgs = []
    for imgF, exampleDng in zip(imgFilesToConvert, exampleDngFiles):
        rimg = raw.imread(exampleDng)
        img = cv2.imread(imgF, flags=cv2.IMREAD_GRAYSCALE)

        inverted = (img / 255) / config.wb_filter_scaled[config.margin_crop:(config.height - config.margin_crop),
                                 config.margin_crop:(config.width - config.margin_crop)] + config.black_level

        inverted = cv2.copyMakeBorder(inverted, config.margin_crop, config.margin_crop, config.margin_crop,
                                      config.margin_crop, cv2.BORDER_REFLECT)
        raw_image = rimg.raw_image
        raw_image[:] = inverted

        rgb = rimg.postprocess()
        rgb = rgb[config.margin_crop:(config.height - config.margin_crop),
              config.margin_crop:(config.width - config.margin_crop)]
        if writeFiles:
            rgbFile = join(outFolder, os.path.basename(imgF) + '.png')
            imageio.imsave(rgbFile, rgb)
        rgbImgs.append(rgb)
        # print(img.shape)

    return rgbImgs

def undistortMultiCamsFolder(inFolder, outFolder, camParamF, extname='pgm'):
    os.makedirs(outFolder, exist_ok=True)

    inFiles = sortedGlob(join(inFolder, '*.' + extname))
    camParams = json.load(open(camParamF))['cam_params']
    for iCam, inF in enumerate(inFiles):
        outFile = join(outFolder, Path(inF).stem + '.png')
        # img = cv2
        preprocessImg(inF, outFile, camParams[str(iCam)])

if __name__ == '__main__':
    # exampleFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03052\A03052.pgm'
    #
    # img = cv2.imread(exampleFile, cv2.IMREAD_GRAYSCALE)
    # # imgBayer= inverseConverGrayImg(img)
    # # cv2.imwrite('ImgBayer.png', imgBayer)
    #
    # imgColor = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
    # cv2.imshow('imgColor', imgColor)
    # cv2.waitKey()

    # inFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\CleanPlateExtracted\gray\distorted'
    # inFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03052'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    inFolder = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\00_FiducialMarkerTrackingPipeline\Data'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    outFolder = join(inFolder, 'RgbUndist')

    undistortMultiCamsFolder(inFolder, outFolder, camParamF)
