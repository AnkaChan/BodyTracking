from S07_ToSilhouetteFitting_MultiFrames import *

def makeComparison(obj_filename, fittedMeshFile, cleanPlateFolder, targetImageFolder, camParamF):
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)
    texture_image = mesh.textures.maps_padded()
    normals = mesh.verts_normals_packed()

    fittedMesh = pv.PolyData(fittedMeshFile)
    pts = torch.tensor(fittedMesh.points, requires_grad=False, device=device)
    mesh = mesh.update_padded(pts[None, ...] / 1000)

    # load background texture
    cp_out, cp_crop_out = load_images(cleanPlateFolder, cropSize=1080, UndistImgs=True, camParamF=camParamF,
                                      imgExt='png')
    refs_out, refs_crops_out = load_images(targetImageFolder, cropSize=1080, UndistImgs=True,
                                           camParamF=camParamF, imgExt='pgm')
    refImgFiles = glob.glob(join(targetImageFolder, '*.pgm'))
    refImgFiles.sort()

    # set up light
    xyz = torch.from_numpy(np.float32([0, 0, 2000]))[None]

    # fixed light brightness
    #         diffuse = 1.2505254745483398
    #         ambient = 0.5083667039871216
    #         specular = 0.0
    diffuse = 0.0
    # ambient = 1.0
    # ambient = 1.0
    ambient = 0.5

    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)

    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device)

    # build renderer
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)
    # rendererSynth = Renderer(device, cfg)

    # Change specular color to green and change material shininess
    # materials = Materials(
    #     device=device,
    #     specular_color=[[0.0, 1.0, 0.0]],
    #     shininess=10.0
    # )
    # plt.imshow(texture_image.cpu().numpy()[0, ..., :3])
    # plt.show()

    os.makedirs(outputFolder, exist_ok=True)
    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            bg = cp_crop_out[iCam]
            bg = torch.tensor(bg[None], dtype=torch.float32, device=device)
            rendererSynth.blend_params = BlendParams(sigma=cfg.sigma, gamma=1e-4, background_color=bg)

            image_cur = rendererSynth.renderer(mesh,
                                               cameras=cams[iCam],
                                               # materials=materials,
                                               blend_params=rendererSynth.blend_params
                                               )
            img = cv2.flip(image_cur.cpu().numpy()[0, ..., :3], -1)
            # plt.imshow(img)
            # plt.show()
            imgP = Path(refImgFiles[iCam])
            cv2.imwrite(join(outputFolder, imgP.stem + '_0Rendered.png'), (255 * img).astype(np.uint8))
            cv2.imwrite(join(outputFolder, imgP.stem + '_1Reference.png'),
                        (255 * cv2.flip(refs_crops_out[iCam], -1)).astype(np.uint8))

            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)


if __name__ == '__main__':
    cfg = RenderingCfg()

    cfg.sigma = 0
    cfg.blurRange = 0
    cfg.plotStep = 20
    cfg.numCams = 16
    # low learning rate for pose optimization
    cfg.learningRate = 1e-3
    cfg.batchSize = 4
    # cfg.faces_per_pixel = 6 # for testing
    cfg.faces_per_pixel = 5  # for debugging
    # cfg.imgSize = 2160
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 0.000001
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 500
    # cfg.numIterations = 20
    cfg.kpFixingWeight = 0

    # # Set paths
    # DATA_DIR = r"..\Data\TextureMap"
    # obj_filename = os.path.join(DATA_DIR, "SMPLWithSocks.obj")
    # fittedMeshFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final\06550\InterpolatedMesh.ply'
    # cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted'
    # targetImageFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06550'
    # outputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\Comparison\06550'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    # makeComparison(obj_filename, fittedMeshFile, cleanPlateFolder, targetImageFolder, camParamF)

    # Set paths
    DATA_DIR = r"..\Data\TextureMap"
    obj_filename = os.path.join(DATA_DIR, "SMPLWithSocks.obj")
    cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    fittedMeshParentFolder = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final'
    targetImageParentFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
    outputParentFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\Comparison'

    inMeshFolders = glob.glob(join(fittedMeshParentFolder, '*'))

    for meshFolder in inMeshFolders:
        fittedMeshFile = join(meshFolder, 'InterpolatedMesh.ply')
        frameName = Path(meshFolder).stem
        targetImageFolder = join(targetImageParentFolder, frameName)
        outputFolder = join(outputParentFolder, frameName)

        makeComparison(obj_filename, fittedMeshFile, cleanPlateFolder, targetImageFolder, camParamF)




