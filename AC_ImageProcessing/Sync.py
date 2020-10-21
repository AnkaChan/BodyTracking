import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal
import shutil
from pathlib import Path

class Config:
    def __init__(self):
        self.peakHeight = 5
        self.numDeadFrameBegin = 50
        self.numDeadFrameEnd = 50
        self.zfillNum = 5
        self.ext = 'pgm'
        self.renameFolder = False
        self.addFolderNameBeforeOutputFile = False


def fileParts(filePath):
    path, f = os.path.split(filePath)
    name, ext = os.path.splitext(f)
    return path, name, ext

def getPeaks(intensityFiles, numDeadFrameBegin, numDeadFrameEnd, peakHeight=5):
    peaksForCams = []
    periodsForCams = []

    for fileName in intensityFiles:
        # print("file: ", fileName)
        meanIntens = np.load(fileName)
        # print(meanIntens)
        # l1 = plt.plot(range(len(meanIntens)), meanIntens, 'ro')
        diffs = np.zeros((len(meanIntens) - 1,), np.float32)
        for i in range(len(meanIntens) - 1):
            diffs[i] = math.fabs(meanIntens[i + 1] - meanIntens[i])

        peaks, _ = scipy.signal.find_peaks(diffs, height=peakHeight)

        peaks = np.delete(peaks, np.where(peaks < numDeadFrameBegin), axis=0)
        peaks = np.delete(peaks, np.where(peaks > len(diffs) - numDeadFrameEnd), axis=0)

        peaksSelected = []
        peakDiffs = []

        for i in range(len(peaks)-1):
            peakDiffs.append(peaks[i+1] - peaks[i])
            peaksSelected

        # print(peaks)
        # print(peakDiffs)

        peaksForCams.append(peaks)
        periodsForCams.append(peakDiffs)

    return peaksForCams, periodsForCams

def checkPeaks(peaksForCams, periodsForCams):
    for i in range(1, len(peaksForCams)):
        periods1 = periodsForCams[i-1]
        periods2 = periodsForCams[i]

        matchLength = min(len(periods1), len(periods2))

        for j in range(matchLength):
            if periods1[j] != periods2[j]:
                return False

    return True

def getSyncPoints(intensityFiles, numDeadFrameBegin = 50, numDeadFrameEnd = 50, peakHeight=5):
    peaksForCams, periodsForCams = getPeaks(intensityFiles, numDeadFrameBegin, numDeadFrameEnd, peakHeight)
    ret = checkPeaks(peaksForCams, periodsForCams)

    if ret:
        print("Sync succeed!")
        syncPoints = []
        for p in peaksForCams:
            syncPoints.append(p[0])

        print("The sync points are: ", syncPoints)
        return syncPoints
    else:
        print("Error in sync! Periods do not match!")
        print("peaksForCams:")
        for p in peaksForCams:
            print(p)
        for peroid in periodsForCams:
            print(peroid)
        print("periodsForCams")
        print(periodsForCams)
        input1 = input("Force sync? Y/n (Forced sync will sync the sequences in first peak)")
        if input1 == 'Y' or input1 == 'y':
            syncPoints = []
            for p in peaksForCams:
                syncPoints.append(p[0])
            print('Apply forced sync.')
            print("The sync points are: ", syncPoints)
            return syncPoints
        else:
            exit(-1)
        return []


def renameAccordingToSequence(inFolder, ext = "pgm", addFolderName=False, zfillNum=5):
    i = 0
    inFiles = glob.glob(inFolder + "/*." + ext)
    folderPath = Path(inFolder)
    inFiles.sort()
    for filename in inFiles:
        src = filename
        dst = str(i)
        if addFolderName:
            dst = inFolder + "/" + folderPath.stem + dst.zfill(zfillNum) + "." + ext
        else:
            dst = inFolder + "/" + dst.zfill(zfillNum) + "." + ext


        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1

def syncOneSequence(sequencePath, syncPoint, ext = "pgm", addFolderName=False, zfillNum=5):
    fileNames = glob.glob(sequencePath + "/*." + ext)
    fileNames.sort()
    filesToDelete = fileNames[0: syncPoint]
    os.makedirs(sequencePath + "/Deleted", exist_ok=True)
    for f in filesToDelete:
        path, name, _ = fileParts(f)
        shutil.move(f, sequencePath + "/Deleted/" + name + "." + ext)

    renameAccordingToSequence(sequencePath, ext, addFolderName, zfillNum)

def syncCameraSequences(inSeqFolders, intensityFiles = [], config = Config()):
    if len(intensityFiles) == 0:
        newInFolders = []
        for folder in inSeqFolders:
            print("Syncing: ", folder)
            folderStemName = Path(folder).stem
            newFolderStemName = folderStemName[:1]
            parentFolder = str(Path(folder).parent)
            newFolderFullPath = os.path.join(parentFolder, newFolderStemName)
            os.rename(folder, newFolderFullPath)

            newInFolders.append(newFolderFullPath)

        inSeqFolders = newInFolders
        for folder in inSeqFolders:
            intensityFiles.append(folder + "/mean_intensities.npy")

    syncPoints = getSyncPoints(intensityFiles, config.numDeadFrameBegin, config.numDeadFrameEnd, config.peakHeight)
    for i in range(len(inSeqFolders)):
        syncOneSequence(inSeqFolders[i], syncPoints[i], config.ext, config.addFolderNameBeforeOutputFile, config.zfillNum)

if __name__ == "__main__":
    ## Used to recover the wrong naming...
    inFolder = r'Z:\2019_12_09_capture\Convert'
    folderNames = ['A', 'N', 'O', 'P']

    for folderName in folderNames:
        inFolderForCame = os.path.join(inFolder, folderName)
        inFiles = glob.glob(inFolderForCame + '\*.pgm')

        inFiles.sort()
        # beginId = 261
        for i, f in enumerate(inFiles):
            dst = inFolder + '/' + inFolderForCame + "/" + inFolderForCame + str(i).zfill(5) + "." + 'pgm'
            os.rename(f, dst)
    exit()
    #
    config = Config()

    # fileName = r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\A001_03082015_C001\mean_intensities.npy"
    # meanIntens = np.load(fileName)
    # print(meanIntens)
    #
    # # l1 = plt.plot(range(len(meanIntens)), meanIntens, 'ro')
    #
    # diffs = np.zeros((len(meanIntens)-1,), np.float32)
    # for i in range(len(meanIntens) - 1):
    #     diffs[i] = math.fabs(meanIntens[i+1] - meanIntens[i])
    #
    # l2 = plt.plot(diffs)
    #
    # peaks, _ =  scipy.signal.find_peaks(diffs, height= 10)
    # plt.plot(peaks, diffs[peaks], "x")
    # plt.plot(np.zeros_like(diffs), "--", color="gray")
    #
    # print(peaks)
    #
    # plt.show()

    # folders = [
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\A001_03082015_C001",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\B001_03280254_C001",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\C001_03251035_C001",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\D001_03050804_C001",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\E001_03060921_C001",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\F001_03290207_C001",
    # ]

    # folders = glob.glob(r"F:\WorkingCopy2\2019_04_16_8CamsCapture\*")

    folders = glob.glob(r"Z:\2019_12_09_capture\Convert\*")
    folders = folders[1:]
    config.renameFolder = True
    config.addFolderNameBeforeOutputFile = True

    # fileNames = [
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\A001_03082015_C001\mean_intensities.npy",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\B001_03280254_C001\mean_intensities.npy",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\C001_03251035_C001\mean_intensities.npy",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\D001_03050804_C001\mean_intensities.npy",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\E001_03060921_C001\mean_intensities.npy",
    #     r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\F001_03290207_C001\mean_intensities.npy",
    # ]

    syncCameraSequences(folders, config=config)

    # syncOneSequence(r"E:\Mocap\WorkingCopy\2019_03_29_6CameNewLightSet\Transformed\Test", 3)
    # Check the sync

    # l2 = plt.plot(diffs)
    # plt.plot(peaks, diffs[peaks], "x")
    # plt.plot(np.zeros_like(diffs), "--", color="gray")
    # plt.show()
