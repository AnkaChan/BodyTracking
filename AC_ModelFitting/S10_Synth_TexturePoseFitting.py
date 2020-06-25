from S07_ToSilhouetteFitting_MultiFrames import *



if __name__ == '__main__':
    cfg = RenderingCfg()

    # cfg.sigma = 0
    # cfg.blurRange = 0
    cfg.sigma = 1e-6
    cfg.blurRange = 1e-6

    cfg.plotStep = 20
    cfg.numCams = 16
    # low learning rate for pose optimization
    cfg.learningRate = 1e-3
    # cfg.learningRate = 1
    # cfg.learningRate = 100

    cfg.batchSize = 4
    cfg.faces_per_pixel = 6 # for testing
    # cfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 500
    # cfg.numIterations = 20
    cfg.kpFixingWeight = 0
    cfg.noiseLevel = 0.1
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    pose_size = 3 * 52
    beta_size = 10

    inputs = InputBundle()

    # Set paths
    DATA_DIR = r"..\Data\TextureMap"
    obj_filename = os.path.join(DATA_DIR, "SMPLWithSocks.obj")
    cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted\Undist'
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    fitParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final\03067\FitParam_Param_00499.npz'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\SynthPoseFitting'

    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    mesh = load_objs_as_meshes([obj_filename], device=device)

    # Read fitting parameter
    transInit, poseInit, betaInit, personalShape = loadCompressedFittingParam(fitParamFile, readPersonalShape=True)
    transInit = transInit / 1000
    personalShape = personalShape / 1000

    # Make fitting parameter tensors
    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=False, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=False, device=device)
    personalShape = torch.tensor(personalShape, dtype=torch.float64, requires_grad=False, device=device)

    # Build up smplsh model
    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=personalShape)

    # Build up the sparse point cloud constraint
    interpoMat = np.load(inputs.toSparsePCMat)
    registeredCornerIds = np.where(np.any(interpoMat, axis=1))[0]
    print("Number of registered corners:", registeredCornerIds.shape)

    # load camera and distort image
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device)
    cp_out, cp_crop_out = load_images(cleanPlateFolder, cropSize=1080, UndistImgs=False, camParamF=camParamF, imgExt='png')
    backgrounds = []
    for iCam in range(len(cams)):
        bg = cp_crop_out[iCam]
        bg = torch.tensor(bg[None], dtype=torch.float32, device=device, requires_grad=False)
        backgrounds.append(bg)

    outFolderForExperiment, outFolderMesh, = makeOutputFolder(inputs.outputFolder, cfg, Prefix='PoseFitting_')
    print('outFolderForExperiment:', outFolderForExperiment)

    # build renderer
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

    with torch.no_grad():
        verts = smplsh(betas, pose, trans).type(torch.float32)
        smplshMesh = mesh.update_padded(verts[None])

    refImages = renderImagesWithBackground(cams, rendererSynth, smplshMesh, backgrounds)

    visualize2DResults(refImages, outImgFile=join(outFolderForExperiment, 'RefImages.png'), withAlpha=False)

    # generate perturbed mesh and visualize it
    np.random.seed(cfg.randSeedPerturb)
    if cfg.bodyJointOnly:
        numParameters = 3 * 22
    else:
        numParameters = 3 * 52
    # posePerturbed = torch.tensor(pose.cpu().numpy() + (np.random.rand(pose_size) - 0.5) * noiseLevel, dtype=torch.float64, device=device, requires_grad=True)
    # Keep hand fixed
    if cfg.bodyJointOnly:
        poseHands = pose[numParameters:].clone().detach()
        poseParams = torch.tensor(
            pose[:numParameters].cpu().detach().numpy() + (np.random.rand(numParameters) - 0.5) * cfg.noiseLevel,
            dtype=torch.float64, device=device, requires_grad=True)
        posePerturbed = torch.cat([poseParams, poseHands])
    else:
        poseParams = torch.tensor(pose.cpu().detach().numpy() + (np.random.rand(pose_size) - 0.5) * cfg.noiseLevel,
                                  dtype=torch.float64, device=device, requires_grad=True)
        posePerturbed = poseParams

    with torch.no_grad():
        vertsPerturbed = smplsh(betas, posePerturbed, trans).type(torch.float32)
        smplshMeshPerturbed = mesh.update_padded(vertsPerturbed[None])

    initialImages = renderImagesWithBackground(cams, rendererSynth, smplshMeshPerturbed, backgrounds)
    visualize2DResults(initialImages, outImgFile=join(outFolderForExperiment, 'Fit00000_initial.png'), withAlpha=False)

    # the optimization loop
    losses = []
    optimizer = torch.optim.Adam([poseParams], lr=cfg.learningRate)

    logFile = join(outFolderForExperiment, 'Logs.txt')
    logger = Logger.configLogger(logFile, )

    loop = tqdm(range(cfg.numIterations))
    fitParamFolder = join(outFolderForExperiment, 'FitParam')
    os.makedirs(fitParamFolder, exist_ok=True)
    diffImageFolder = join(outFolderForExperiment, 'DiffImage')
    os.makedirs(diffImageFolder, exist_ok=True)

    # main optimization loop
    for i in loop:
        optimizer.zero_grad()

        lossVal = 0
        for iCam in range(cfg.numCams):
            if cfg.bodyJointOnly:
                #         poseHands = pose[numBodyParameters:].clone().detach()
                posePerturbed = torch.cat([poseParams, poseHands])
            else:
                posePerturbed = poseParams

            blend_params = BlendParams(
                rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam])
            refImg = torch.tensor(refImages[iCam,..., :3], dtype=torch.float32, device=device, requires_grad=False)
            verts = smplsh(betas, posePerturbed, trans).type(torch.float32)
            smplshMesh = mesh.update_padded(verts[None])

            images = rendererSynth.renderer(smplshMesh, cameras=cams[iCam], blend_params=blend_params)
            # there are some NaN pixels, this is a walk around
            images[images != images] = 0
            refImg[refImg != refImg] = 0
            loss = torch.norm(refImg - images[0, ..., :3], p=1)
            loss = torch.mean(torch.abs(refImg - images[0, ..., :3]))

            loss.backward()
            lossVal += loss.item()

        if cfg.bodyJointOnly:
            #         poseHands = pose[numBodyParameters:].clone().detach()
            posePerturbed = torch.cat([poseParams, poseHands])
        else:
            posePerturbed = poseParams
        # joint regularizer
        loss = cfg.jointRegularizerWeight * torch.sum((posePerturbed ** 2))
        loss.backward()
        jointRegularizerLoss = loss.item()

        # recordData
        losses.append(lossVal)

        optimizer.step()

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'image loss %.6f, jointRegularizerLoss %.6f, MemUsed:%.2f' \
                  % (lossVal, jointRegularizerLoss, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        # if lossVal < cfg.terminateLoss:
        #    break

        # Save outputs to create a GIF.
        if (i + 1) % cfg.plotStep == 0:
            showCudaMemUsage(device)
            with torch.no_grad():
                verts = smplsh(betas, posePerturbed, trans).type(torch.float32)
                smplshMesh = mesh.update_padded(verts[None])

            plt.close('all')

            outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
            images = renderImagesWithBackground(cams, rendererSynth, smplshMesh, backgrounds)

            outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
            np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
                     beta=betas.cpu().detach().numpy(), personalShape=personalShape.cpu().detach().numpy())
            visualize2DResults(images, outImgFile=join(outFolderForExperiment, outImgFile), withAlpha=False)

            diffImgs = np.stack([np.abs(img - refImg) for img, refImg in zip(images, refImages)])
            outDiffImgFile = join(diffImageFolder, 'Fig_' + str(i).zfill(5) + '.png')
            visualize2DResults(diffImgs, outImgFile=outDiffImgFile, withAlpha=False)
            # saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
            #         smplshExampleMesh)









