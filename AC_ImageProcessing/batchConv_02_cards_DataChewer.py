import numpy as np
import rawpy as raw
import cv2
import glob
import os
import time
from multiprocessing import Pool
import tqdm

import argparse

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
    #brightness_mult = 5.6
    brightness_mult = 7

    # folder that contains sub-directories with DNG images; example folder name is "A001_01011738_C008"
    # src_folder = r"d:\mocap\2019_01_10_FourCamera"
    # src_folder = "D:/Mocap/01_25_2019FourCam"
    # src_folder = r"E:\Mocap\WorkingCopy\2019_02_22_5CamReconstruction"
    # src_fold er = r"E:\Mocap\WorkingCopy\2019_03_06_6CamCapture"
    src_folders = []    

    # folder that will contain the output sub-directories with converted images:
    # dst_folder = r"E:\Mocap\WorkingCopy\2019_02_22_5CamReconstruction\Transformed"
    dst_folder = ""
    # either -1 to process all files, or a cap on total number of images to process in each folder (good for testing):
    #max_frames = -1
    startId = 0
    max_frames = -1

    # how many worker processes to launch in parallel
    num_processes = 16

    framesSelected = None

def convert_file(inf_outf_tuple):
    config, infname, outfname = inf_outf_tuple
#    print(infname)
    with raw.imread(infname) as rimg:
        assert(rimg.raw_image.shape[0] == config.height and rimg.raw_image.shape[1] == config.width)
        clip_black = np.clip(rimg.raw_image, config.black_level, config.white_level) - config.black_level
        fbright = 255*np.clip(clip_black * config.wb_filter_scaled, 0, 1)[config.margin_crop:(config.height-config.margin_crop),config.margin_crop:(config.width-config.margin_crop)]
        cv2.imwrite(outfname, fbright)
        mean_intensity = np.mean(fbright)
    return mean_intensity

def batchConvert(config = Configs()):

    # src_files = glob.glob(src_mask)
    # if __name__ == '__main__':
    #     print("Main process: Found %i input files" % len(src_files))
    # else:
    #     print("Worker process starting")

    # prepare scaling matrix (wb_filter_scaled)
    M1 = np.array([[1,0],[0,0]])
    M2 = np.array([[0,1],[0,0]])
    M3 = np.array([[0,0],[1,0]])
    M4 = np.array([[0,0],[0,1]])
    filter1 = np.tile(M1, (int(config.height/2), int(config.width/2)))
    filter2 = np.tile(M2, (int(config.height/2), int(config.width/2)))
    filter3 = np.tile(M3, (int(config.height/2), int(config.width/2)))
    filter4 = np.tile(M4, (int(config.height/2), int(config.width/2)))

    config.wb_filter_scaled = config.brightness_mult * (filter1*config.wb_scales[0] + filter2*config.wb_scales[1] + filter3*config.wb_scales[2] + filter4) / (config.white_level - config.black_level)

    print("The following folders are going to be processed one by one using %i processes:" % config.num_processes)
    os.makedirs(config.dst_folder, exist_ok=True)

    for img_fold in config.src_folders:
        print(img_fold)

    outFolders = []

    if config.framesSelected is None:
        for img_fold in config.src_folders:
            print("processing folder: %s" % img_fold)
            src_files = glob.glob(img_fold + r"\*.dng")
            folder_name = os.path.basename(img_fold)
            out_folder = os.path.join(config.dst_folder, folder_name)
            outFolders.append(out_folder)
            os.makedirs(out_folder, exist_ok=True)
            num_files = len(src_files)
            if config.max_frames != -1:
                num_files = min(num_files, config.max_frames)
            print("Creating folder %s and converting there %i out of %i files" % (out_folder, num_files, len(src_files)))
            conv_args = []

            for fname in src_files[config.startId:num_files]:
                conv_args.append((config, fname, os.path.join(config.dst_folder, folder_name, os.path.basename(fname) + ".pgm")))

            #print(conv_args)
            start = time.time()
            with Pool(config.num_processes) as p:
                mean_intensities = p.map(convert_file, conv_args) #, chunksize=1)
            end = time.time()
            print("Elapsed time: ", end - start)
            np.save(os.path.join(out_folder, "mean_intensities"), mean_intensities)
        return outFolders

    else:
        for i, iFrame in enumerate(config.framesSelected):
            out_folder = os.path.join(config.dst_folder, str(i).zfill((8)))
            print("processing folder: %s" % str(iFrame).zfill((8)))
            conv_args = []

            os.makedirs(out_folder, exist_ok=True)
            for img_fold in config.src_folders:
                src_files = glob.glob(img_fold + r"\*.dng")
                # folder_name = os.path.basename(img_fold)

                fname = src_files[iFrame]
                conv_args.append(
                    (config, fname, os.path.join(out_folder, os.path.basename(fname) + ".pgm")))

            with Pool(config.num_processes) as p:
                mean_intensities = p.map(convert_file, conv_args) #, chunksize=1)


if __name__ == "__main__":
    # srcFolders = [
    #     r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\A",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\B",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\C",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\D",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\E",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\F",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\G",
    #     # r"E:\Mocap\WorkingCopy\2019_06_03_NewSuitCapture\H",
    # ]

    #srcFolders = glob.glob(r'H:\SharedData\2019_11_28_16CamsCapture\Raw\*')
    # srcFolders = [
    #     'E:\G001_06160601_C002',
    # ]

    #outFolder = r'Z:\shareZ\2019_12_13_Lada_Capture\Converted'
    #outFolder =  r'Z:\shareZ\2020_01_01_KateyCapture\Converted'
    outFolder =  r'Z:\shareZ\2020_03_22_NewSuitDesignCapture\Converted'
    parser = argparse.ArgumentParser(description='Batch Convert')
    parser.add_argument(
        '--check',
        default=-1,
        help='Number of frames used to check'
    )
    my_namespace = parser.parse_args()

    # numFrameToCheck = my_namespace.check

    srcFolders = []
    
    if len(glob.glob(r'D:\*001')):
        srcFolders.append(glob.glob(r'D:\*001')[0])
    if len(glob.glob(r'E:\*001')):    
        srcFolders.append(glob.glob(r'E:\*001')[0])
    if len(glob.glob(r'F:\*001')):
        srcFolders.append(glob.glob(r'F:\*001')[0])
    if len(glob.glob(r'G:\*001')):
        srcFolders.append(glob.glob(r'G:\*001')[0])
    if len(glob.glob(r'H:\*001')):
        srcFolders.append(glob.glob(r'H:\*001')[0])
    if len(glob.glob(r'I:\*001')):
        srcFolders.append(glob.glob(r'I:\*001')[0])
    if len(glob.glob(r'J:\*001')):
        srcFolders.append(glob.glob(r'J:\*001')[0])
    if len(glob.glob(r'K:\*001')):
        srcFolders.append(glob.glob(r'K:\*001')[0])

    #assert(len(srcFolders) == 8)

    # srcFolders = glob.glob(r'Z:\2019_11_28_16CamsCapture\Raw\*')

    config = Configs()


    if int(my_namespace.check) > 0:
        numToCheck = int(my_namespace.check)
        outFolder = outFolder + r'\Check'

        print("Checking conversion using %d frames." % numToCheck)
        minLength = -1
        for img_fold in srcFolders:
            # print("processing folder: %s" % img_fold)
            src_files = glob.glob(img_fold + r"\*.dng")
            if len(src_files) < minLength or minLength < 0:
                minLength = len(src_files)

        # indices = list(range(minLength))
        # np.random.shuffle(indices)
        # indices = indices[:numToCheck]
        indices = [int(i * (minLength-1) / numToCheck) for i in range(numToCheck)]
        print("Frames selected: ", indices)

        config.framesSelected = indices


    os.makedirs(outFolder, exist_ok=True)
    config.src_folders = srcFolders
    config.dst_folder = outFolder

    batchConvert(config)
