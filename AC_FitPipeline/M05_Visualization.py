import  os, json
from os.path import join
from pathlib import Path
import pyvista as pv
from Utility import *
from SkelFit import Visualization

# print(len(vns))
# print(len(vs))
# print(len(vts))
# print(len(fs))

def loadObjWithUV(vt_path):
    vts = []
    fs = []
    vns = []
    vs = []
    with open(vt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l = line.split(' ')
            if l[0] == 'vt':
                u = l[1]
                v = l[2].split('\n')[0]
                vts.append([u, v])
            elif l[0] == 'vn':
                vns.append([0, 0, 0])
            elif l[0] == 'v':
                vs.append([0, 0, 0])
            elif l[0] == 'f':
                fs_curr = []
                for i in range(len(l) - 1):
                    fi = l[i + 1].split('/')
                    fi = '{}/{}/{}'.format(fi[0], fi[1], fi[2].split('\n')[0])
                    fs_curr.append(fi)
                fs.append(fs_curr)
        f.close()

    return vs, vts, vns, fs
def converObjsInFolder(obj_dir, out_dir, vt_path = r'..\Data\TextureMap2Color\SMPLWithSocks_tri.obj'):
    os.makedirs(out_dir, exist_ok=True)

    vs, vts, vns, fs = loadObjWithUV(vt_path)

    in_paths = glob.glob(obj_dir + '/*.obj')
    for in_path in in_paths:
        obj_name = Path(in_path).stem
        out_path = out_dir + '/{}.obj'.format(obj_name)

        convertObjFile(in_path, out_path, vts, fs, vns)


def convertObjFile(inFile, outFile, vts, fs, vns):
    # obj_name = in_path.split('\\')[-1]

    extName = Path(inFile).suffix
    # read current
    vs = []

    if extName.lower() == '.obj':
        with open(inFile, 'r') as f:
            lines = f.readlines()

            for line in lines:
                l = line.split(' ')
                if l[0] == 'v':
                    vs.append([l[1], l[2], l[3].split('\n')[0]])
                elif l[0] == 'vt' or l[0] == 'vn':
                    assert (False)
            f.close()
    else:
        mesh = pv.PolyData(inFile)
        vs = mesh.points.tolist()

    # write new
    with open(outFile, 'w+') as f:
        for i, v in enumerate(vs):
            vn = vns[i]
            f.write('vn {} {} {}\n'.format(vn[0], vn[1], vn[2]))
            f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
        for vt in vts:
            f.write('vt {} {}\n'.format(vt[0], vt[1]))
        for face in fs:
            f.write('f')
            for fi in face:
                f.write(' {}'.format(fi))
            f.write('\n')
        f.close()
    print(outFile)
    print(len(vs))


def visualizeToSparseFitting(toSparseFittingFolder, outFolder=None, addUV=False, objWithUV=r'..\Data\TextureMap2Color\SMPLWithSocks_tri.obj'):
    if outFolder is None:
        outFolder = join(toSparseFittingFolder, 'Vis')

    os.makedirs(outFolder, exist_ok=True)

    frameFolders = sortedGlob(join(toSparseFittingFolder, '*'))

    if addUV:
        outFolderObjWithUV = join(outFolder, 'ObjWithUV')
        os.makedirs(outFolderObjWithUV, exist_ok=True)
        vs, vts, vns, fs = loadObjWithUV(objWithUV)
        for frameFolder in frameFolders:
            frameName = Path(frameFolder).stem
            deformedObjFile = join(frameFolder, 'ToSparseMesh.obj')

            outObj = join(outFolderObjWithUV, 'A' + frameName + '.obj')

            convertObjFile(deformedObjFile, outObj, vts, fs, vns)

            mesh = pv.PolyData(outObj)
            mesh.save(join(outFolder, 'A' + frameName + '.ply'))
    else:
        for frameFolder in frameFolders:
            frameName = Path(frameFolder).stem

            deformedObjFile = join(frameFolder, 'ToSparseMesh.obj')
            mesh = pv.PolyData(deformedObjFile)
            mesh.save(join(outFolder, 'A' + frameName + '.ply'))





if __name__ == '__main__':
    # toSparseFittignFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    # visualizeToSparseFitting(toSparseFittignFolder, addUV=True)

    kpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'
    # visualizeToSparseFitting(kpFolder, addUV=False)
    Visualization.obj2vtkFolder(kpFolder)

    sparseMeshFOdler = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LadaStand'
    # visualizeToSparseFitting(sparseMeshFOdler, addUV=False)
    Visualization.obj2vtkFolder(sparseMeshFOdler)

    pass
