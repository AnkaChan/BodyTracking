# from S14_TexturedFitting import *
import shutil
import glob, os, tqdm
from os.path import join

if __name__ == '__main__':
    inFolder = r'Z:\shareZ\2020_06_30_AC_ConsequtiveTexturedFitting2\PerVertexFitting'
    outFolder = r'Z:\shareZ\2020_06_30_AC_ConsequtiveTexturedFitting2\Final'

    fittingFolders = glob.glob(join(inFolder, '*'))
    os.makedirs(outFolder, exist_ok=True)
    for iFrame in tqdm.tqdm(range(len(fittingFolders))):
        frameFolder = fittingFolders[iFrame]
        frameDataFolder = glob.glob(join(frameFolder, '*'))[-1]
        meshFile =  glob.glob(join(frameDataFolder, 'Mesh', '*.ply'))[-1]

        shutil.copy(meshFile, join(outFolder, 'A' + os.path.basename(frameFolder) + '.ply'))




