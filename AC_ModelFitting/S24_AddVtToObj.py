import glob
import os
from pathlib import Path
import pyvista as pv

vt_path = r'..\Data\TextureMap\SMPLWithSocks_tri.obj'
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
                fi = l[i+1].split('/')
                fi = '{}/{}/{}'.format(fi[0], fi[1], fi[2].split('\n')[0])
                fs_curr.append(fi)
            fs.append(fs_curr)
    f.close()
# print(len(vns))
# print(len(vs))
# print(len(vts))
# print(len(fs))


def converObjsInFolder(obj_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    in_paths = glob.glob(obj_dir + '/*.obj')
    for in_path in in_paths:
        obj_name = Path(in_path).stem
        out_path = out_dir + '/{}.obj'.format(obj_name)

        convertObjFile(in_path, out_path)


def convertObjFile(inFile, outFile):
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

if __name__ == '__main__':
    # obj_dir = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\IniitalTexture\Meshes'
    # # out_dir = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\FinalObj\WithTextureCoord'
    # out_dir = os.path.join(obj_dir, 'WithTextureCoord')
    # converObjsInFolder(obj_dir, out_dir)

    inFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolatedWithSparse.ply'
    outFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\FinalMesh.obj'
    convertObjFile(inFile, outFile)