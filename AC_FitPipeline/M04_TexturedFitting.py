from Config import *
from Utility_Rendering import *
from copy import copy
from tqdm import tqdm
import shutil


def makeOutputFolder(outputParentFolder, cfg, Prefix = ''):
    expName = Prefix + 'Sig' + str(cfg.sigma) + '_BR' + str(cfg.blurRange) + '_Fpp' + str(
        cfg.faces_per_pixel) \
              + '_NCams' + str(cfg.numCams) + '_ImS' + str(cfg.imgSize) + '_LR' + str(cfg.learningRate) + '_JR' + str(
        cfg.jointRegularizerWeight) + '_KPW' + str(cfg.kpFixingWeight) + '_SCW' + str(cfg.toSparseCornersFixingWeight) \
        + '_It' + str(cfg.numIterations)

    outFolderForExperiment = join(outputParentFolder, expName)
    os.makedirs(outFolderForExperiment, exist_ok=True)

    json.dump({"CfgSynth": cfg.__dict__ },
              open(join(outFolderForExperiment, 'cfg.json'), 'w'), indent=2)

    outFolderMesh = join(outFolderForExperiment, 'Mesh')
    os.makedirs(outFolderMesh, exist_ok=True)

    return outFolderForExperiment, outFolderMesh,

def texturedPoseFitting(inputs, cfg, device, ):
    if cfg.makeOutputSubfolder:
        outFolderForExperiment, outFolderMesh = makeOutputFolder(inputs.outputFolder, cfg, 'Pose_')
    else:
        outFolderForExperiment = inputs.outputFolder
        outFolderMesh = join(outFolderForExperiment, 'Mesh')
        os.makedirs(outFolderMesh, exist_ok=True)

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
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize, withoutExtrinsics=cfg.extrinsicsOutsideCamera)

    # load Images
    image_refs_out, crops_out = load_images(inputs.imageFolder, camParamF=inputs.camParamF, UndistImgs=cfg.undistImg, cropSize=cfg.imgSize, imgExt=cfg.inputImgExt)
    crops_out = np.stack(crops_out, axis=0)
    cp_out, cp_crop_out = load_images(inputs.cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=inputs.camParamF,
                                      imgExt='png')

    if cfg.drawInitial:
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

    sparsePC = torch.tensor(sparsePC, dtype=torch.float32, requires_grad=False, device=device)
    interpoMat = torch.tensor(interpoMat, dtype=torch.float32, requires_grad=False, device=device)

    # load texture
    mesh = load_objs_as_meshes([inputs.texturedMesh], device=device)
    # textureMp = np.squeeze(mesh.textures.maps_padded().cpu().numpy())
    # cv2.imshow('Texture', textureMp)
    # cv2.waitKey()

    # load pose
    if inputs.compressedStorage:
        transInit, poseInit, betaInit, initialPersonalShape = loadCompressedFittingParam(inputs.initialFittingParamFile, readPersonalShape=True)
        # transInit = transInit
        xyzShift = torch.tensor(initialPersonalShape, dtype=torch.float32, requires_grad=False, device=device)
    else:
        transInit = np.load(inputs.initialFittingParamTranslationFile)
        poseInit = np.load(inputs.initialFittingParamPoseFile)
        betaInit = np.load(inputs.initialFittingParamBetasFile)

        initialPersonalShape = np.load(inputs.personalShapeFile)
        xyzShift = torch.tensor(initialPersonalShape, dtype=torch.float32, requires_grad=False, device=device)

    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=xyzShift, unitMM=False)

    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=True, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=True, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=True, device=device)

    # verts = smplsh(betas, pose, trans).type(torch.float32)
    # smplshMesh = Meshes([verts], [smplsh.faces.to(device)])
    with torch.no_grad():
        verts = smplsh(betas, pose, trans).type(torch.float32)
        smplshMesh = mesh.update_padded(verts[None])

    print('outFolderForExperiment: ', outFolderForExperiment)

    # set up light
    xyz = torch.from_numpy(np.float32([0, 0, 2000]))[None]
    diffuse = 0.0
    ambient = cfg.ambientLvl
    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)

    # initial image
    if cfg.drawInitial:
        meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds, cams_torch=cams_torch, cfg=cfg)
        visualize2DResults(images, outImgFile=join(outFolderForExperiment, 'Fig_00000_Initial.png'), withAlpha=False, sizeInInches=5)

    # initial diff image
    diffImageFolder = join(outFolderForExperiment, 'DiffImage')
    os.makedirs(diffImageFolder, exist_ok=True)
    if cfg.drawInitial:
        diffImgs = np.stack([np.abs(img[...,:3] - refImg) for img, refImg in zip(images, crops_out)])
        # print("Initial loss:", loss)
        visualize2DResults(diffImgs, outImgFile=join(diffImageFolder, 'Fig_00000_Initial.png'), sizeInInches=5)

    # the optimization loop
    if cfg.optimizerType == 'Adam':
        optimizer = torch.optim.Adam([trans, pose, betas], lr=cfg.learningRate)
    elif cfg.optimizerType == 'SGD':
        optimizer = torch.optim.SGD([trans, pose, betas], lr=cfg.learningRate)

    losses = []
    toSparseCloudLosses = []
    headKpFixingLosses = []

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
        verts = smplsh(betas, pose, trans).type(torch.float32)
        # smplshMesh = mesh.update_padded(verts[None])
        smplshMesh = mesh.offset_verts(verts - mesh.verts_packed())

        meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        for iCam in range(len(cams)):
            meshesTransformed = updataMeshes(meshes, cams_torch, iCam, cfg)
            blend_params = BlendParams(
                rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam])
            images = rendererSynth.renderer(meshesTransformed, cameras=cams[iCam], blend_params=blend_params)
            images[images != images] = 0.5
            # images[torch.isnan(images)] = 0.5
            # refImg[refImg != refImg] = 0
            loss = torch.mean(torch.abs(imagesBatchRefs[iCam] - images[..., :3]))
            loss.backward(retain_graph=True)
            lossVal += loss.item()

        # joint regularizer
        loss = cfg.jointRegularizerWeight * torch.sum((pose ** 2))
        loss.backward()

        # to corners loss
        verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
        loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        loss.backward()
        #     lossVal += loss.item()
        toSparseCloudLoss = loss.item()

        # to keypoint fixing loss
        # verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
        # headJoints = smplshRegressorMatHead @ verts
        # loss = cfg.kpFixingWeight * torch.sum((headJoints - headKps) ** 2)
        # loss.backward()
        # headKpFixingLoss = loss.item()
        headKpFixingLoss = 0

        losses.append(lossVal)
        toSparseCloudLosses.append(toSparseCloudLoss)
        headKpFixingLosses.append(headKpFixingLoss)

        optimizer.step()

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'image loss %.6f, toSparseCloudLoss %.6f, headKpFixingLoss %.4f, MemUsed:%.2f' \
                  % (lossVal, toSparseCloudLoss, headKpFixingLoss, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        terminate = False
        # if lossVal < cfg.terminateLoss:
        #    break
        if i>cfg.errAvgLength:
            avgStep = np.mean(np.abs(np.array(losses[-11:-1]) - np.array(losses[-10:])))
            if avgStep < cfg.terminateStep:
                logger.info("Teminate because average step length in " + str(cfg.errAvgLength) + "steps is: " + str(
                    avgStep) + " less than: " + str(cfg.terminateStep))

                terminate = True

        # Save outputs to create a GIF.
        if (i + 1) % cfg.plotStep == 0 or terminate or (i==0 and cfg.drawInitial):
            lossesFile = join(outFolderForExperiment, 'Errs.json')
            json.dump({'ImageLoss':losses, 'toSparseCloudLosses':toSparseCloudLosses, 'headKpFixingLosses':headKpFixingLosses}, open(lossesFile, 'w'))

            showCudaMemUsage(device)
            with torch.no_grad():
                verts = smplsh(betas, pose, trans).type(torch.float32)
                smplshMesh = mesh.update_padded(verts[None])
                meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])

            plt.close('all')

            outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
            images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds, cams_torch=cams_torch, cfg=cfg)

            outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
            np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
                     beta=betas.cpu().detach().numpy(), personalShape=xyzShift.cpu().detach().numpy())
            visualize2DResults(images, outImgFile=join(outFolderForExperiment, outImgFile), withAlpha=False, sizeInInches=5)

            diffImgs = np.stack([np.abs(img[...,:3] - refImg) for img, refImg in zip(images, crops_out)])
            outDiffImgFile = join(diffImageFolder, 'Fig_' + str(i).zfill(5) + '.png')
            visualize2DResults(diffImgs, outImgFile=outDiffImgFile, withAlpha=False, sizeInInches=5)
            saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

            # make comparison view
            comparisonFolderThisIter = join(comprisonFolder, str(i).zfill(5) )
            os.makedirs(comparisonFolderThisIter, exist_ok=True)
            for iCam in range(images.shape[0]):
                img = (cv2.flip(images[iCam, ...,:3], -1) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_0Rendered.png' ), img)
                imgRef = (cv2.flip(crops_out[iCam, ...], -1)*255).astype(np.uint8)
                imgRef = cv2.cvtColor(imgRef, cv2.COLOR_RGB2BGR)
                cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_1Ref.png'), imgRef)

            # draw loss curve
            fig, a_loss  = plt.subplots()
            a_loss.plot(losses, linewidth=3)
            a_loss.set_title('losses: {}'.format(losses[-1]))
            a_loss.grid()
            fig.savefig(join(outFolderForExperiment, 'ErrCurve_'+ cfg.optimizerType + '_LR' +str(cfg.learningRate)+'_TStep' +str(cfg.terminateStep) + '.png'),
                        dpi=256, transparent=False, bbox_inches='tight', pad_inches=0)
        if terminate:
            break

def texturedPerVertexFitting(inputs, cfg, device):
    outFolderForExperiment = inputs.outputFolder
    outFolderMesh = join(inputs.outputFolder, 'Mesh')
    os.makedirs(outFolderMesh, exist_ok=True)

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
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize, withoutExtrinsics=cfg.extrinsicsOutsideCamera)

    # load Images
    image_refs_out, crops_out = load_images(inputs.imageFolder, camParamF=inputs.camParamF, UndistImgs=cfg.undistImg,
                                            cropSize=cfg.imgSize, imgExt=cfg.inputImgExt)
    crops_out = np.stack(crops_out, axis=0)
    cp_out, cp_crop_out = load_images(inputs.cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False,
                                      camParamF=inputs.camParamF,
                                      imgExt='png')

    if cfg.drawInitial:
        refImgsFolder = join(outFolderForExperiment, 'RefImages')
        os.makedirs(refImgsFolder, exist_ok=True)
        for iCam in range(len(crops_out)):
            cv2.imwrite(join(refImgsFolder, 'A' + str(iCam).zfill(5) + '.png'),
                        (255 * crops_out[iCam]).astype(np.uint8))

    backgrounds = []
    for iBatch in range(len(cams)):
        bg = cp_crop_out[iBatch * cfg.batchSize:(iBatch + 1) * cfg.batchSize:]
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
        transInit, poseInit, betaInit, initialPersonalShape = loadCompressedFittingParam(inputs.initialFittingParamFile,
                                                                                         readPersonalShape=True)
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
    ambient = cfg.ambientLvl
    specular = 0.0
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)

    if cfg.drawInitial:
        # initial image
        meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds, cams_torch=cams_torch, cfg=cfg)
        visualize2DResults(images, outImgFile=join(outFolderForExperiment, 'Fig_00000_Initial.png'), withAlpha=False,
                           sizeInInches=5)

    # initial diff image
    diffImageFolder = join(outFolderForExperiment, 'DiffImage')
    os.makedirs(diffImageFolder, exist_ok=True)

    if cfg.drawInitial:
        diffImgs = np.stack([np.abs(img[..., :3] - refImg) for img, refImg in zip(images, crops_out)])
        # print("Initial loss:", loss)
        visualize2DResults(diffImgs, outImgFile=join(diffImageFolder, 'Fig_00000_Initial.png'), sizeInInches=5)

    # the optimization loop
    if cfg.optimizePose:
        # optimizer = torch.optim.Adam([xyzShift, trans, pose, betas], lr=cfg.learningRate)
        optimizer = torch.optim.SGD([xyzShift, trans, pose, betas], lr=cfg.learningRate)
    else:
        # optimizer = torch.optim.Adam([xyzShift], lr=cfg.learningRate)
        optimizer = torch.optim.SGD([xyzShift, trans, pose, betas], lr=cfg.learningRate)

    losses = []
    toSparseCloudLosses = []
    lpSmootherLosses = []

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
    if cfg.optimizerType == 'Adam':
        optimizer = torch.optim.Adam([xyzShift], lr=cfg.learningRate)
    elif cfg.optimizerType == 'SGD':
        optimizer = torch.optim.SGD([xyzShift], lr=cfg.learningRate)

    # main optimization loop
    for i in loop:
        optimizer.zero_grad()

        lossVal = 0
        verts = smplsh(betas, pose, trans).type(torch.float32)
        # smplshMesh = mesh.update_padded(verts[None])
        smplshMesh = mesh.offset_verts(verts - mesh.verts_packed())

        meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
        for iCam in range(len(cams)):
            meshesTransformed = updataMeshes(meshes, cams_torch, iCam, cfg)
            blend_params = BlendParams(
                rendererSynth.blend_params.sigma, rendererSynth.blend_params.gamma, background_color=backgrounds[iCam])
            images = rendererSynth.renderer(meshesTransformed, cameras=cams[iCam], blend_params=blend_params)
            images[images != images] = 0.5
            # images[torch.isnan(images)] = 0.5
            # refImg[refImg != refImg] = 0
            loss = torch.mean(torch.abs(imagesBatchRefs[iCam] - images[..., :3]))
            loss.backward(retain_graph=True)
            # loss.backward(keep_graph= True)

            lossVal += loss.item()

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
        # lossVal += loss.item()
        # to corners loss
        verts = smplsh(betas, pose, trans).type(torch.float32)
        loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        loss.backward()
        toSparseCloudLoss = loss.item()

        # fixing loss
        loss = torch.sum(xyzShift[indicesToFix, :] ** 2)
        loss.backward()
        headKpFixingLoss = loss.item()

        optimizer.step()

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        losses.append(lossVal)
        toSparseCloudLosses.append(toSparseCloudLoss)
        lpSmootherLosses.append(lpSmootherVal)

        infoStr = 'Fitting loss %.6f, normal regularizer loss %.6f, Laplacian regularizer loss %.6f, toSparseCloudLoss %.6f, MemUsed:%.2f' \
                  % (lossVal, normalSmootherVal, lpSmootherVal, toSparseCloudLoss, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        terminate = False

        if i > cfg.errAvgLength:
            avgStep = np.mean(np.abs(np.array(losses[-11:-1]) - np.array(losses[-10:])))
            if avgStep < cfg.terminateStep:
                logger.info("Teminate because average step length in " + str(cfg.errAvgLength) + "steps is: " + str(
                    avgStep) + " less than: " + str(cfg.terminateStep))
                terminate = True

        # Save outputs to create a GIF.
        if (i + 1) % cfg.plotStep == 0 or terminate or (i==0 and cfg.drawInitial):
            showCudaMemUsage(device)
            lossesFile = join(outFolderForExperiment, 'Errs.json')
            json.dump({'ImageLoss': losses, 'toSparseCloudLosses': toSparseCloudLosses,
                       'LaplacianSmootherLoss': lpSmootherLosses}, open(lossesFile, 'w'))

            with torch.no_grad():
                verts = smplsh(betas, pose, trans).type(torch.float32)
                smplshMesh = mesh.update_padded(verts[None])
                meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])

            plt.close('all')

            outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
            images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds, cams_torch=cams_torch, cfg=cfg)

            outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
            np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
                     beta=betas.cpu().detach().numpy(), personalShape=xyzShift.cpu().detach().numpy())
            visualize2DResults(images, outImgFile=join(outFolderForExperiment, outImgFile), withAlpha=False,
                               sizeInInches=5)

            diffImgs = np.stack([np.abs(img[..., :3] - refImg) for img, refImg in zip(images, crops_out)])
            outDiffImgFile = join(diffImageFolder, 'Fig_' + str(i).zfill(5) + '.png')
            visualize2DResults(diffImgs, outImgFile=outDiffImgFile, withAlpha=False, sizeInInches=5)
            saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

            # make comparison view
            comparisonFolderThisIter = join(comprisonFolder, str(i).zfill(5))
            os.makedirs(comparisonFolderThisIter, exist_ok=True)
            for iCam in range(images.shape[0]):
                img = (cv2.flip(images[iCam, ..., :3], -1) * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_0Rendered.png'), img)
                imgRef = (cv2.flip(crops_out[iCam, ...], -1) * 255).astype(np.uint8)
                imgRef = cv2.cvtColor(imgRef, cv2.COLOR_RGB2BGR)
                cv2.imwrite(join(comparisonFolderThisIter, str(iCam).zfill(5) + '_1Ref.png'), imgRef)

            # draw loss curve
            fig, a_loss  = plt.subplots()
            a_loss.plot(losses, linewidth=3)
            a_loss.set_title('losses: {}'.format(losses[-1]))
            a_loss.grid()
            fig.savefig(join(outFolderForExperiment, 'ErrCurve_'+ cfg.optimizerType + '_LR' +str(cfg.learningRate)+'_TStep' +str(cfg.terminateStep) + '.png'),
                        dpi=256, transparent=False, bbox_inches='tight', pad_inches=0)

        if terminate:
            break

