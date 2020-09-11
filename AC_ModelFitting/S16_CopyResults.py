# from S14_TexturedFitting import *
import shutil
# import Utility
import glob, os, tqdm
from os.path import join
import pyvista as pv


def write_obj(verts, file_name):
    # with open(file_name, 'w') as fp:
    #   for v in verts:
    #     fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    #
    #   for f in self.faces + 1:
    #     fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))
    with open(file_name, 'w') as fp:
        for v in verts:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))


if __name__ == '__main__':
    inFolder = r'Z:\shareZ\2020_06_30_AC_ConsequtiveTexturedFitting2\PerVertexFitting'
    outFolder = r'Z:\shareZ\2020_06_30_AC_ConsequtiveTexturedFitting2\Final'
    outFolderObj = r'Z:\shareZ\2020_06_30_AC_ConsequtiveTexturedFitting2\FinalObj'

    fittingFolders = glob.glob(join(inFolder, '*'))
    os.makedirs(outFolder, exist_ok=True)
    os.makedirs(outFolderObj, exist_ok=True)
    for iFrame in tqdm.tqdm(range(len(fittingFolders))):
        frameFolder = fittingFolders[iFrame]
        frameDataFolder = glob.glob(join(frameFolder, '*'))[-1]
        meshFile = glob.glob(join(frameDataFolder, 'Mesh', '*.ply'))[-1]

        shutil.copy(meshFile, join(outFolder, 'A' + os.path.basename(frameFolder) + '.ply'))

        mesh = pv.PolyData(meshFile)
        write_obj(mesh.points, join(outFolderObj, 'A' + os.path.basename(frameFolder) + '.obj'))



