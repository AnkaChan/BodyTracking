from S07_ToSilhouetteFitting_MultiFrames import *
import time

def texturedPerVertexFitting(inputs, cfg, device):
    outFolderForExperiment, outFolderMesh, = makeOutputFolder(inputs.outputFolder, cfg, Prefix='Pose_')

    handIndices = json.load(open(inputs.handIndicesFile))
    headIndices = json.load(open(inputs.HeadIndicesFile))

    indicesToFix = copy(handIndices)
    indicesToFix.extend(headIndices)

    smplshExampleMesh = pv.PolyData(inputs.smplshExampleMeshFile)
    nVerts = smplshExampleMesh.points.shape[0]

    LNP = np.load('SmplshRestposeLapMat.npy')

    BiLNP = LNP @ LNP
    if cfg.biLaplacian:
        LNP = torch.tensor(BiLNP, dtype=torch.float32, device=device, requires_grad=False)
    else:
        LNP = torch.tensor(LNP, dtype=torch.float32, device=device, requires_grad=False)

    # normalShift = torch.tensor(np.full((nVerts,1), 0), dtype=torch.float32, requires_grad = True, device=device)


    # load cameras
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize)

    # load Images
    image_refs_out, crops_out = load_images(inputs.imageFolder, camParamF=inputs.camParamF, UndistImgs=True, cropSize=cfg.imgSize, imgExt='pgm')
    crops_out = np.stack(crops_out, axis=0)
    cp_out, cp_crop_out = load_images(inputs.cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=inputs.camParamF,
                                      imgExt='png')

    refImgsFolder = join(outFolderForExperiment, 'RefImages')
    os.makedirs(refImgsFolder, exist_ok=True)
    for iCam in range(len(crops_out)):
        cv2.imwrite(join(refImgsFolder, 'A' + str(iCam).zfill(5) + '.png'), (255*crops_out[iCam]).astype(np.uint8))

    backgrounds = []
    for iBatch in range(len(cams)):
        bg = cp_crop_out[iBatch*cfg.batchSize:(iBatch+1)*cfg.batchSize:]
        bg = torch.tensor(bg, dtype=torch.float32, device=device, requires_grad=False)
        backgrounds.append(bg)

    # Build up the sparse point cloud constraint
    interpoMat = np.load(inputs.toSparsePCMat)

    registeredCornerIds = np.where(np.any(interpoMat, axis=1))[0]
    print("Number of registered corners:", registeredCornerIds.shape)

    sparsePC = pv.PolyData(inputs.sparsePointCloudFile)
    sparsePC = np.array(sparsePC.points)

    constraintIds = np.where(sparsePC[:, 2] > 0)[0]
    constraintIds = np.intersect1d(registeredCornerIds, constraintIds)
    print("Number of constraint corners:", constraintIds.shape)

    interpoMat = interpoMat[constraintIds, :]
    sparsePC = sparsePC[constraintIds, :]
    # initial to sparse point cloud dis

    sparsePC = torch.tensor(sparsePC, dtype=torch.float32, requires_grad=False, device=device) / 1000
    interpoMat = torch.tensor(interpoMat, dtype=torch.float32, requires_grad=False, device=device)

    # load texture
    mesh = load_objs_as_meshes([inputs.texturedMesh], device=device)

    # load pose
    if inputs.compressedStorage:
        transInit, poseInit, betaInit, initialPersonalShape = loadCompressedFittingParam(inputs.initialFittingParamFile, readPersonalShape=True)
        # transInit = transInit
    else:
        transInit = np.load(inputs.initialFittingParamTranslationFile)
        poseInit = np.load(inputs.initialFittingParamPoseFile)
        betaInit = np.load(inputs.initialFittingParamBetasFile)
        initialPersonalShape = np.load(inputs.personalShapeFile) / 1000

    xyzShift = torch.tensor(initialPersonalShape, dtype=torch.float32, requires_grad=True, device=device)
    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=xyzShift, unitMM=False)

    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=cfg.optimizePose, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=cfg.optimizePose, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=cfg.optimizePose, device=device)

    # verts = smplsh(betas, pose, trans).type(torch.float32)
    # smplshMesh = Meshes([verts], [smplsh.faces.to(device)])
    with torch.no_grad():
        verts = smplsh(betas, pose, trans).type(torch.float32)
        smplshMesh = mesh.update_padded(verts[None])

    print('outFolderForExperiment: ', outFolderForExperiment)

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

    # initial image
    meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
    images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds)
    # visualize2DResults(images, outImgFile=join(outFolderForExperiment, 'Fig_00000_Initial.png'), withAlpha=False, sizeInInches=5)

    # initial diff image
    diffImageFolder = join(outFolderForExperiment, 'DiffImage')
    os.makedirs(diffImageFolder, exist_ok=True)

    diffImgs = np.stack([np.abs(img[...,:3] - refImg) for img, refImg in zip(images, crops_out)])
    # print("Initial loss:", loss)
    # visualize2DResults(diffImgs, outImgFile=join(diffImageFolder, 'Fig_00000_Initial.png'), sizeInInches=5)

    # the optimization loop
    if cfg.optimizePose:
        optimizer = torch.optim.Adam([xyzShift, trans, pose, betas], lr=cfg.learningRate)
    else:
        optimizer = torch.optim.Adam([xyzShift], lr=cfg.learningRate)

    losses = []
    logFile = join(outFolderForExperiment, 'Logs.txt')
    logger = Logger.configLogger(logFile, )

    fitParamFolder = join(outFolderForExperiment, 'FitParam')
    os.makedirs(fitParamFolder, exist_ok=True)
    comprisonFolder = join(outFolderForExperiment, 'Comparison')
    os.makedirs(comprisonFolder, exist_ok=True)

    imagesBatchRefs = []
    for iCam in range(len(cams)):
        imagesBatchRef = crops_out[iCam * cfg.batchSize:iCam * cfg.batchSize + cfg.batchSize, ...]
        imagesBatchRef = torch.tensor(imagesBatchRef, dtype=torch.float32, device=device, requires_grad=False)
        imagesBatchRefs.append(imagesBatchRef)

    loop = tqdm(range(cfg.numIterations))
    # main optimization loop
    for i in loop:
        optimizer.zero_grad()

        lossVal = 0
        timeRender = time.clock()
        for iCam in range(len(cams)):

            verts = smplsh(betas, pose, trans).type(torch.float32)
            smplshMesh = mesh.update_padded(verts[None])

            meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
            blend_params = BlendParams(
                rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam])
            images = rendererSynth.renderer(meshes, cameras=cams[iCam], blend_params=blend_params)
            images[images != images] = 0
            # refImg[refImg != refImg] = 0
            loss = torch.mean(torch.abs(imagesBatchRefs[iCam] - images[..., :3]))
            loss.backward()
            lossVal += loss.item()

        timeRender = time.clock() - timeRender

        timeRegularizer = time.clock()
        # joint regularizer
        if cfg.optimizePose:
            loss = cfg.jointRegularizerWeight * torch.sum((pose ** 2))
            loss.backward()

        # verts = smplsh(betas, pose, trans).type(torch.float32)
        # smplshMesh = mesh.update_padded(verts[None])
        #     loss = cfg.lpSmootherW * mesh_laplacian_smoothing(mesh) + cfg.normalSmootherW * mesh_normal_consistency(mesh)
        # loss = cfg.normalSmootherW * mesh_normal_consistency(smplshMesh)
        # normalSmootherVal = loss.item()
        normalSmootherVal = 0
        loss = 0
        loss = loss + cfg.lpSmootherW * xyzShift[:, 0:1].transpose(0, 1) @ LNP @ xyzShift[:, 0:1]
        loss = loss + cfg.lpSmootherW * xyzShift[:, 1:2].transpose(0, 1) @ LNP @ xyzShift[:, 1:2]
        loss = loss + cfg.lpSmootherW * xyzShift[:, 2:3].transpose(0, 1) @ LNP @ xyzShift[:, 2:3]
        lpSmootherVal = loss.item() - normalSmootherVal

        loss.backward()
        timeRegularizer = time.clock() - timeRegularizer
        # lossVal += loss.item()
        # to corners loss

        timeToSparse = time.clock()
        verts = smplsh(betas, pose, trans).type(torch.float32)
        loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        loss.backward()
        toSparseCloudLoss = loss.item()

        # fixing loss
        loss = torch.sum(xyzShift[indicesToFix, :] ** 2)
        loss.backward()
        headKpFixingLoss = loss.item()
        timeToSparse = time.clock() - timeToSparse

        timeOptStep = time.clock()
        optimizer.step()
        timeOptStep = time.clock() - timeOptStep

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'timeRender %.6f, timeRegularizer %.6f, LtimeToSparse %.6f, timeOptStep %.6f, MemUsed:%.2f' \
                  % (timeRender, timeRegularizer, timeToSparse, timeOptStep, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        # # Save outputs to create a GIF.
        # if (i + 1) % cfg.plotStep == 0:
        #     showCudaMemUsage(device)
        #     with torch.no_grad():
        #         verts = smplsh(betas, pose, trans).type(torch.float32)
        #         smplshMesh = mesh.update_padded(verts[None])
        #         meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        #
        #     plt.close('all')
        #
        #     outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
        #     images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds)
        #
        #     outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
        #     np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
        #              beta=betas.cpu().detach().numpy(), personalShape=xyzShift.cpu().detach().numpy())
        #     visualize2DResults(images, outImgFile=join(outFolderForExperiment, outImgFile), withAlpha=False, sizeInInches=5)
        #
        #     diffImgs = np.stack([np.abs(img[...,:3] - refImg) for img, refImg in zip(images, crops_out)])
        #     outDiffImgFile = join(diffImageFolder, 'Fig_' + str(i).zfill(5) + '.png')
        #     visualize2DResults(diffImgs, outImgFile=outDiffImgFile, withAlpha=False, sizeInInches=5)
        #     saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
        #             smplshExampleMesh)
        #
        #     # make comparison view
        #     comparisonFolderThisIter = join(comprisonFolder, str(i).zfill(5) )
        #     os.makedirs(comparisonFolderThisIter, exist_ok=True)
        #     for iCam in range(images.shape[0]):
        #         img = (cv2.flip(images[iCam, ...,:3], -1) * 255).astype(np.uint8)
        #         cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_0Rendered.png' ), img)
        #         imgRef = (cv2.flip(crops_out[iCam, ...], -1)*255).astype(np.uint8)
        #         cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_1Ref.png'), imgRef)
#
if __name__ == '__main__':
    cfg = RenderingCfg()

    # cfg.sigma = 0
    # cfg.blurRange = 0
    cfg.sigma = 1e-8
    cfg.blurRange = 1e-8

    cfg.plotStep = 10000
    cfg.numCams = 16
    # low learning rate for pose optimization
    # cfg.learningRate = 2e-3
    cfg.learningRate = 1e-4
    # cfg.learningRate = 1
    # cfg.learningRate = 100

    cfg.batchSize = 2
    # cfg.faces_per_pixel = 10 # for testing
    cfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    cfg.imgSize = 1080
    cfg.terminateLoss = 0.1
    # cfg.lpSmootherW = 1e-10
    # cfg.lpSmootherW = 1e-4
    cfg.lpSmootherW = 1e-1
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    cfg.numIterations = 500
    # cfg.numIterations = 20
    cfg.useKeypoints = False
    cfg.kpFixingWeight = 0
    cfg.noiseLevel = 0.1
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    cfg.toSparseCornersFixingWeight = 1
    cfg.bin_size = None
    # cfg.cull_backfaces = True

    pose_size = 3 * 52
    beta_size = 10

    inputs = InputBundle()

    inputs.imageFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06950'
    # inputs.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
    inputs.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\06950\A00006950.obj'
    inputs.cleanPlateFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\CleanPlatesExtracted\gray\distorted\Undist'
    inputs.compressedStorage = True
    inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_21_TextureRendering\Model\06950\Param_00959.npz'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\RealDataPerVertsFitting\06950_TimeTest'
    # copy all the final result to this folder
    # inputs.finalOutputFolder = r'F:\WorkingCopy2\2020_06_21_TextureRendering\RealDataPoseFitting'
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    texturedPerVertexFitting(inputs, cfg, device)










