import  os, json
from os.path import join
from pathlib import Path
import pyvista as pv
from Utility import *
from Utility_Rendering import *
from Config import *
from SkelFit import Visualization
import tqdm

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

def renderFrame(meshFile, inTextureMeshFile, camParamF, outFolder, cleanPlateFolder=None,  cfg=RenderingCfg(),
                            convertToM=False, rendererType='RGB'):
    camNames = ['A', 'B', 'C', 'D', 'E', "F", 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # load cameras
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize)

    # load clean plate
    if cleanPlateFolder is not None:
        cp_out, cp_crop_out = load_images(cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=camParamF,
                                      imgExt='png')
        backgrounds = []
        for iBatch in range(len(cams)):
            bg = cp_crop_out[iBatch * cfg.batchSize:(iBatch + 1) * cfg.batchSize:]
            bg = torch.tensor(bg, dtype=torch.float32, device=device, requires_grad=False)
            backgrounds.append(bg)
    else:
        backgrounds = None

    # load textured mesh
    mesh = load_objs_as_meshes([inTextureMeshFile], device=device)

    # set up light
    xyz = torch.from_numpy(np.float32([0, 0, 2000]))[None]
    diffuse = 0.0
    ambient = cfg.ambientLvl
    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)

    if rendererType == 'RGB':
        rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)
    elif rendererType == 'Silhouette':
        rendererSynth = Renderer(device,  cfg=cfg)
    else:
        assert False, 'Unknow renderer type:' + rendererType

    deformedMesh = pv.PolyData(meshFile)
    if convertToM:
        deformedMesh.points = deformedMesh.points / 1000

    with torch.no_grad():
        verts = torch.tensor(deformedMesh.points, dtype=torch.float32, device=device, requires_grad=False)
        smplshMesh = mesh.update_padded(verts[None])

        # render image
        meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        images = []
        for iCam in range(len(cams)):
            # if device is not None:
            #     showCudaMemUsage(device)
            blend_params = BlendParams(
                rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam] if backgrounds is not None else None)
            image_cur = rendererSynth.renderer(meshes, cameras=cams[iCam], blend_params=blend_params)

            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)

    # save rendered images
    for iCam in range(len(camNames)):
        outRenderedFile = join(outFolder, join(camNames[iCam] + Path(meshFile).stem + '.png'))
        if rendererType == 'RGB':
            img = images[iCam,...,:3]
        elif rendererType == 'Silhouette':
            img = images[iCam,...,3]

        img = cv2.flip(img, -1)

        imageio.imsave(outRenderedFile, (255*img).astype(np.uint8))


def renderConsecutiveFrames(inFramesFolder, cleanPlateFolder, inTextureMeshFile, camParamF, outFolder, frameNames=None, cfg=RenderingCfg(),
                            inMeshExt='ply', convertToM=False, rendererType='RGB'):
    camNames = ['A', 'B', 'C', 'D', 'E', "F", 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    if frameNames is None:
        inDeformedFiles = glob.glob(join(inFramesFolder, '*.') + inMeshExt)
    else:
        inDeformedFiles = [join(inFramesFolder, 'A' + frameName + '.' + inMeshExt) for frameName in frameNames]

    # load clean plate
    cp_out, cp_crop_out = load_images(cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=camParamF,
                                      imgExt='png')

    # load cameras
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize)

    backgrounds = []
    for iBatch in range(len(cams)):
        bg = cp_crop_out[iBatch*cfg.batchSize:(iBatch+1)*cfg.batchSize:]
        bg = torch.tensor(bg, dtype=torch.float32, device=device, requires_grad=False)
        backgrounds.append(bg)

    # load textured mesh
    mesh = load_objs_as_meshes([inTextureMeshFile], device=device)

    # set up light
    xyz = torch.from_numpy(np.float32([0, 0, 2000]))[None]
    diffuse = 0.0
    ambient = cfg.ambientLvl
    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)

    if rendererType == 'RGB':
        rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)
    elif rendererType == 'Silhouette':
        rendererSynth = Renderer(device,  cfg=cfg)
    else:
        assert False, 'Unknow renderer type:' + rendererType

    renderedOutFolders = []
    for i in range(len(camNames)):
        camName = camNames[i]
        renderedOutFolder = join(outFolder, camName, 'Rendered')
        os.makedirs(renderedOutFolder, exist_ok=True)
        renderedOutFolders.append(renderedOutFolder)

    loop = tqdm.tqdm(range(len(inDeformedFiles)))
    for iFrame in loop:
        inDeformedMeshFile = inDeformedFiles[iFrame]
        deformedMesh = pv.PolyData(inDeformedMeshFile)
        if convertToM:
            deformedMesh.points = deformedMesh.points / 1000

        with torch.no_grad():
            verts = torch.tensor(deformedMesh.points, dtype=torch.float32, device=device, requires_grad=False)
            smplshMesh = mesh.update_padded(verts[None])

            # render image
            meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
            images = []
            for iCam in range(len(cams)):
                # if device is not None:
                #     showCudaMemUsage(device)
                blend_params = BlendParams(
                    rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam])
                image_cur = rendererSynth.renderer(meshes, cameras=cams[iCam], blend_params=blend_params)

                images.append(image_cur.cpu().detach().numpy())
            images = np.concatenate(images, axis=0)

        # save rendered images
        for iCam in range(len(camNames)):
            if frameNames is not None:
                outRenderedFile = join(renderedOutFolders[iCam], join(frameNames[iFrame] + '.png'))
            else:
                outRenderedFile = join(renderedOutFolders[iCam], join(Path(inDeformedMeshFile).stem + '.png'))
            if rendererType == 'RGB':
                img = images[iCam,...,:3]
            elif rendererType == 'Silhouette':
                img = images[iCam,...,3]

            img = cv2.flip(img, -1)

            imageio.imsave(outRenderedFile, (255*img).astype(np.uint8))



if __name__ == '__main__':
    toSparseFittignFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    visualizeToSparseFitting(toSparseFittignFolder, addUV=True)

    # toSparseFittignFolder = r'F:\WorkingCopy2\2020_08_26_TexturedFitting_LadaGround\ToSparse'
    # finalFittingFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Final\Mesh'
    # converObjsInFolder(finalFittingFolder, join(finalFittingFolder, 'ObjWithUV'), ext='ply', convertToMM=True)


    # kpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'
    # # visualizeToSparseFitting(kpFolder, addUV=False)
    # Visualization.obj2vtkFolder(kpFolder)
    #
    # sparseMeshFOdler = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LadaStand'
    # # visualizeToSparseFitting(sparseMeshFOdler, addUV=False)
    # Visualization.obj2vtkFolder(sparseMeshFOdler)

    pass
