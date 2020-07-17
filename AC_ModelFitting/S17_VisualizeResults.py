from S12_RealData_TexturePerVertexFitting import *


def renderConsecutiveFrames(inFramesFolder, cleanPlateFolder, inTextureMeshFile, camParam, outFolder, cfg=RenderingCfg()):
    camNames = ['A', 'B', 'C', 'D', 'E', "F", 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    inDeformedFiles = glob.glob(join(inFramesFolder, '*.ply'))

    # load clean plate
    cp_out, cp_crop_out = load_images(cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=inputs.camParamF,
                                      imgExt='png')

    # load cameras
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(camParam, device, actual_img_shape, unitM=True)
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
    ambient = 0.5
    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)

    renderedOutFolders = []
    for i in range(len(camNames)):
        camName = camNames[i]
        renderedOutFolder = join(outFolder, camName, 'Rendered')
        os.makedirs(renderedOutFolder, exist_ok=True)
        renderedOutFolders.append(renderedOutFolder)


    loop = tqdm(range(len(inDeformedFiles)))
    for iFrame in loop:
        inDeformedMeshFile = inDeformedFiles[iFrame]
        deformedMesh = pv.PolyData(inDeformedMeshFile)

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
            outRenderedFile = join(renderedOutFolders[iCam], join(Path(inDeformedMeshFile).stem + '.png'))
            img = images[iCam,...,:3]
            img = cv2.flip(img, -1)

            imageio.imsave(outRenderedFile, (255*img).astype(np.uint8))


if __name__ == '__main__':
    inputs = InputBundle()
    inFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Final'
    cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted\Undist'
    outFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\RenderedResult'
    imgParentFolder = r'E:\WorkingCopy\2020_06_30_AC_ConsequtiveTexturedFitting2\Copied\Images'

    cfg = RenderingCfg()

    cfg.sigma = 1e-9
    cfg.blurRange = 1e-9

    cfg.numCams = 16
    cfg.learningRate = 1e-4
    cfg.batchSize = 2
    cfg.faces_per_pixel = 1  # for debugging
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 1e-1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 500
    cfg.useKeypoints = False
    cfg.kpFixingWeight = 0
    cfg.noiseLevel = 0.1
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    cfg.toSparseCornersFixingWeight = 1
    cfg.bin_size = 256

    renderConsecutiveFrames(inFolder, cleanPlateFolder, inputs.texturedMesh, inputs.camParamF, outFolder, cfg=cfg)

    # copy reference images
    imageFolders = glob.glob(join(imgParentFolder, '*'))
    camNames = ['A', 'B', 'C', 'D', 'E', "F", 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    referenceOutFolders = []
    renderedImgFolders = []
    renderedImgFiles = []
    flipComparisonFolders = []
    sideBySideComparisonFolders = []


    for i in range(len(camNames)):
        camName = camNames[i]
        refOutFolder = join(outFolder, camName, 'Reference')
        os.makedirs(refOutFolder, exist_ok=True)
        referenceOutFolders.append(refOutFolder)

        renderedOutFolder = join(outFolder, camName, 'Rendered')
        renderedImgFolders.append(renderedOutFolder)
        renderedImgFiles.append(glob.glob(join(renderedOutFolder, '*.png')))

        flipComparisonFolder = join(outFolder, camName, 'FlipComparison')
        os.makedirs(flipComparisonFolder, exist_ok=True)
        flipComparisonFolders.append(flipComparisonFolder)

        sideBySideComparisonFolder = join(outFolder, camName, 'SideBySideComparison')
        os.makedirs(sideBySideComparisonFolder, exist_ok=True)
        sideBySideComparisonFolders.append(sideBySideComparisonFolder)

    for iFrame, imgFolder in tqdm(enumerate(imageFolders), desc='Copy reference images.'):
        imgFiles = glob.glob(join(imgFolder, '*.pgm'))
        imgFiles.sort()
        # for iCam, imgF in enumerate(imgFiles):
        #     shutil.copy(imgF, join(referenceOutFolders[iCam], os.path.basename(imgF)))
        image_refs_out, crops_out = load_images(imgFolder, camParamF=inputs.camParamF, UndistImgs=True,
                                                cropSize=cfg.imgSize, imgExt='pgm', writeUndistorted=False, normalize=False, flipImg=False)
        for iCam, imgF in enumerate(imgFiles):
            cv2.imwrite(join(referenceOutFolders[iCam], os.path.basename(imgF) + '.png'), crops_out[iCam].astype(np.uint8))

            # make flip comparison
            shutil.copy(renderedImgFiles[iCam][iFrame], join(flipComparisonFolders[iCam], Path(imgF).stem + '.0Rendered.png'))
            cv2.imwrite(join(flipComparisonFolders[iCam], os.path.basename(imgF) + '.png'), crops_out[iCam].astype(np.uint8))

            # make side by side comparison
            collageImage = np.concatenate([cv2.imread(renderedImgFiles[iCam][iFrame]), crops_out[iCam].astype(np.uint8)], axis=1)
            # cv2.imshow('sideBySideComparison', collageImage)
            # cv2.waitKey()
            cv2.imwrite(join(sideBySideComparisonFolders[iCam], os.path.basename(imgF) + '.png'), collageImage)






    # generate comparison view and stitched view
    # renderedOutFolders = glob.glob(join(outFolder, 'Rendered'))
    # renderedOutFolders.sort()
    #
    # for renderedCamFolder, referenceCamFolder in zip(renderedOutFolder, referenceOutFolders):
    #     renderedImgFiles = glob.glob(renderedOutFolder)
