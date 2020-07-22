from Config import *
from Utility import *
from copy import copy
from tqdm import tqdm
import shutil

class InputBundle:
    def __init__(s):
        # same over all frames
        s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.smplshRegressorMatFile = r'C:\Code\MyRepo\ChbCapture\08_CNNs\Openpose\SMPLSHAlignToAdamWithHeadNoFemurHead\smplshRegressorNoFlatten.npy'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"
        s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

        # frame specific inputs
        s.imageFolder = r'F:\WorkingCopy2\2020_06_04_SilhouetteExtraction\03067\silhouettes'
        s.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
        s.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\A00003067.obj'

        s.compressedStorage = True
        s.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03067.npz'
        s.outputFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\03067'
        # copy all the final result to this folder
        s.finalOutputFolder = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final'

from S05_InterpolateWithSparsePointCloud import interpolateWithSparsePointCloudSoftly

def loadCompressedFittingParam(file, readPersonalShape=False):
    fitParam = np.load(file)
    transInit = fitParam['trans']
    poseInit = fitParam['pose']
    betaInit = fitParam['beta']

    if readPersonalShape:
        personalShape = fitParam['personalShape']
        return transInit, poseInit, betaInit, personalShape
    else:
        return transInit, poseInit, betaInit

def makeOutputFolder(outputParentFolder, cfg, Prefix = ''):
    expName = Prefix + 'Sig' + str(cfg.sigma) + '_BR' + str(cfg.blurRange) + '_Fpp' + str(
        cfg.faces_per_pixel) \
              + '_NCams' + str(cfg.numCams) + '_ImS' + str(cfg.imgSize) + '_LR' + str(cfg.learningRate) + '_JR' + str(
        cfg.jointRegularizerWeight) + '_KPW' + str(cfg.kpFixingWeight) + '_SCW' + str(cfg.toSparseCornersFixingWeight) \
        # + '_It' + str(cfg.numIterations)

    outFolderForExperiment = join(outputParentFolder, expName)
    os.makedirs(outFolderForExperiment, exist_ok=True)

    json.dump({"CfgSynth": cfg.__dict__ },
              open(join(outFolderForExperiment, 'cfg.json'), 'w'), indent=2)

    outFolderMesh = join(outFolderForExperiment, 'Mesh')
    os.makedirs(outFolderMesh, exist_ok=True)

    return outFolderForExperiment, outFolderMesh,

def renderImages(cams, renderer, mesh):
    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            image_cur = renderer.renderer(mesh, cameras=cams[iCam])
            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)
    # showCudaMemUsage(device)
    return images


def renderImagesWithBackground(cams, renderer, mesh, backgrounds, device=None):
    images = []
    with torch.no_grad():
        for iCam in range(len(cams)):
            if device is not None:
                showCudaMemUsage(device)
            blend_params = BlendParams(
                renderer.blend_params.sigma, renderer.blend_params.gamma, background_color=backgrounds[iCam])
            image_cur = renderer.renderer(mesh, cameras=cams[iCam], blend_params=blend_params)

            images.append(image_cur.cpu().detach().numpy())
        images = np.concatenate(images, axis=0)
        # showCudaMemUsage(device)
    return images


def saveMesh(outFile, verts, smplshExampleMesh):
    smplshExampleMesh.points = verts
    smplshExampleMesh.save(outFile)

def getDiffImages(images, crops_out, cams, cfg):
    diffImages = None
    loss = 0
    for iCam in range(len(cams)):
        imagesBatchRef = crops_out[iCam * cfg.batchSize:iCam * cfg.batchSize + cfg.batchSize, ..., 0]
        imagesBatch = images[iCam * cfg.batchSize:iCam * cfg.batchSize + cfg.batchSize, ..., 3]
        imgDif = np.abs(imagesBatch - imagesBatchRef)
        if diffImages is not None:
            diffImages = np.concatenate([diffImages, imgDif], axis=0)
        else:
            diffImages = imgDif
        loss += 1 - np.sum(np.abs(imagesBatch * imagesBatchRef)) / np.sum(
            np.abs(imagesBatchRef + imagesBatch - imagesBatchRef * imagesBatch))
    return diffImages, loss

def visualize2DSilhouetteResults(images, backGroundImages=None, outImgFile=None, rows=2,
                                 sizeInInches=2):
    numCams = len(images)
    numCols = int(numCams / rows)
    fig, axs = plt.subplots(rows, numCols)
    fig.set_size_inches(numCols * sizeInInches, rows * sizeInInches)
    with torch.no_grad():
        for iRow in range(rows):
            for iCol in range(numCols):
                iCam = rows * iRow + iCol
                imgAlpha = images[iCam, ..., 3]

                if backGroundImages is not None:
                    img = np.copy(backGroundImages[iCam]) * 0.5
                    #                     fgMask = np.logical_not(np.where())
                    #                     for iChannel in range(3):
                    img[..., 0] = img[..., 0] + imgAlpha * 0.5
                    imgAlpha = img

                imgAlpha = cv2.flip(imgAlpha, -1)

                axs[iRow, iCol].imshow(imgAlpha, vmin=0.0, vmax=1.0)
                axs[iRow, iCol].axis('off')

        if outImgFile is not None:
            fig.savefig(outImgFile, dpi=512, transparent=True, bbox_inches='tight', pad_inches=0)


def visualize2DResults(images, backGroundImages=None, outImgFile=None, rows=2, sizeInInches=2, withAlpha=True):
    lossVal = 0
    numCams = len(images)
    numCols = int(numCams / rows)
    fig, axs = plt.subplots(rows, numCols)
    fig.set_size_inches(numCols * sizeInInches, rows * sizeInInches)
    with torch.no_grad():
        for iRow in range(rows):
            for iCol in range(numCols):
                iCam = rows * iRow + iCol
                imgAlpha = images[iCam]

                if backGroundImages is not None:
                    img = np.copy(backGroundImages[iCam])
                    #                     fgMask = np.logical_not(np.where())
                    for iChannel in range(3):
                        img[..., iChannel] = np.where(imgAlpha, imgAlpha, backGroundImages[iCam][..., iChannel])
                    imgAlpha = img

                imgAlpha = cv2.flip(imgAlpha, -1)
                if not withAlpha:
                    imgAlpha = imgAlpha[...,:3]

                axs[iRow, iCol].imshow(imgAlpha, vmin=0.0, vmax=1.0)
                axs[iRow, iCol].axis('off')

        if outImgFile is not None:
            fig.savefig(outImgFile, dpi=512, transparent=True, bbox_inches='tight', pad_inches=0)

def toSilhouettePoseFitting(inputs, cfg):
    pose_size = 3 * 52
    beta_size = 10
    OPHeadKeypoints = [0, 15, 16, 17, 18]
    smplshExampleMesh = pv.PolyData(inputs.smplshExampleMeshFile)

    # The head joint regressor and keypoint data
    Keypoints = pv.PolyData(inputs.KeypointsFile)
    smplshRegressorMat = np.load(inputs.smplshRegressorMatFile)
    smplshRegressorMatHead = smplshRegressorMat[-5:, :]
    smplshRegressorMatHead = torch.tensor(smplshRegressorMatHead, dtype=torch.float32, device=device,
                                          requires_grad=False)
    headKps = torch.tensor(Keypoints.points[OPHeadKeypoints, :], dtype=torch.float32, device=device,
                           requires_grad=False)

    # Read fitting parameter
    if inputs.compressedStorage:
        transInit, poseInit, betaInit = loadCompressedFittingParam(inputs.initialFittingParamFile)
    else:
        transInit = np.load(inputs.initialFittingParamTranslationFile)
        poseInit = np.load(inputs.initialFittingParamPoseFile)
        betaInit = np.load(inputs.initialFittingParamBetasFile)
    personalShape = np.load(inputs.personalShapeFile)
    # Make fitting parameter tensors
    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=True, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=True, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=True, device=device)
    personalShape = torch.tensor(personalShape / 1000, dtype=torch.float64, requires_grad=False, device=device)

    # Build up smplsh model
    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=personalShape)
    verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
    smplshMesh = Meshes([verts], [smplsh.faces.to(device)])

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

    # load camera and distort image
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape)
    cams = init_camera_batches(cams_torch, device)
    image_refs_out, crops_out = load_images(inputs.imageFolder, cropSize=1080, UndistImgs=True, camParamF=inputs.camParamF)
    outFolderForExperiment, outFolderMesh, = makeOutputFolder(inputs.outputFolder, cfg, Prefix='PoseFitting_')
    print('outFolderForExperiment:', outFolderForExperiment)

    # build renderer
    rendererSynth = Renderer(device, cfg)

    # initial image
    images = renderImages(cams, rendererSynth, smplshMesh, )
    visualize2DSilhouetteResults(images, backGroundImages = crops_out, outImgFile=join(outFolderForExperiment, 'Fit0_Initial.png'))

    losses = []
    optimizer = torch.optim.Adam([trans, pose, betas], lr=cfg.learningRate)

    logFile = join(outFolderForExperiment, 'Logs.txt')
    logger = Logger.configLogger(logFile, )

    loop = tqdm(range(cfg.numIterations))
    fitParamFolder = join(outFolderForExperiment, 'FitParam')
    os.makedirs(fitParamFolder, exist_ok=True)

    # main optimization loop
    for i in loop:
        optimizer.zero_grad()

        lossVal = 0
        for iCam in range(cfg.numCams):
            refImg = torch.tensor(crops_out[iCam][..., 0], dtype=torch.float32, device=device, requires_grad=False)
            verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
            smplshMesh = Meshes([verts], [smplsh.faces.to(device)])

            images = rendererSynth.renderer(smplshMesh, cameras=cams[iCam])
            loss = 1 - torch.norm(refImg * images[..., 3], p=1) / torch.norm(
                refImg + images[..., 3] - refImg * images[..., 3], p=1)

            loss.backward()
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

        # recordData
        verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
        headJoints = smplshRegressorMatHead @ verts
        loss = cfg.kpFixingWeight * torch.sum((headJoints - headKps) ** 2)
        loss.backward()
        headKpFixingLoss = loss.item()

        losses.append(lossVal)

        if i:
            optimizer.step()

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'image loss %.6f, toSparseCloudLoss %.6f, headKpFixingLoss %.4f, MemUsed:%.2f' \
                  % (lossVal, toSparseCloudLoss, headKpFixingLoss, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        # if lossVal < cfg.terminateLoss:
        #    break

        # Save outputs to create a GIF.
        if (i+1) % cfg.plotStep == 0:
            showCudaMemUsage(device)
            verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
            smplshMesh = Meshes([verts], [smplsh.faces.to(device)])

            plt.close('all')

            outImgFile = join(outFolderForExperiment, 'Fig_' + str(i).zfill(5) + '.png')
            renderedImages = renderImages(cams, rendererSynth, smplshMesh, )

            outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
            np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
                     beta=betas.cpu().detach().numpy(), personalShape=personalShape.cpu().detach().numpy())
            visualize2DSilhouetteResults(renderedImages, backGroundImages=crops_out, outImgFile=outImgFile,
                                         sizeInInches=5)

            saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

def toSilhouettePerVertexFitting(inputs, cfg):
    handIndices = json.load(open(inputs.handIndicesFile))
    headIndices = json.load(open(inputs.HeadIndicesFile))

    indicesToFix = copy(handIndices)
    indicesToFix.extend(headIndices)

    smplshExampleMesh = pv.PolyData(inputs.smplshExampleMeshFile)
    nVerts = smplshExampleMesh.points.shape[0]

    # LNP = getLaplacian(inputs.smplshExampleMeshFile)
    # np.save('SmplshRestposeLapMat.npy', LNP)

    LNP = np.load('SmplshRestposeLapMat.npy')

    BiLNP = LNP @ LNP
    if cfg.biLaplacian:
        LNP = torch.tensor(BiLNP, dtype=torch.float32, device=device, requires_grad=False)
    else:
        LNP = torch.tensor(LNP, dtype=torch.float32, device=device, requires_grad=False)

    # normalShift = torch.tensor(np.full((nVerts,1), 0), dtype=torch.float32, requires_grad = True, device=device)
    initialPersonalShape = np.load(inputs.personalShapeFile)
    xyzShift = torch.tensor(initialPersonalShape, dtype=torch.float32, requires_grad=True, device=device)

    # load Images
    image_refs_out, crops_out = load_images(inputs.imageFolder, cropSize=cfg.imgSize)
    crops_out = np.stack(crops_out, axis=0)

    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape)

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

    # load pose
    if inputs.compressedStorage:
        transInit, poseInit, betaInit = loadCompressedFittingParam(inputs.initialFittingParamFile)
        transInit = transInit * 1000
    else:
        transInit = np.load(inputs.initialFittingParamTranslationFile) * 1000
        poseInit = np.load(inputs.initialFittingParamPoseFile)
        betaInit = np.load(inputs.initialFittingParamBetasFile)

    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=xyzShift, unitMM=True)

    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=cfg.optimizePose, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=cfg.optimizePose, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=cfg.optimizePose, device=device)

    verts = smplsh(betas, pose, trans).type(torch.float32)
    smplshMesh = Meshes([verts], [smplsh.faces.to(device)])

    outFolderForExperiment, outFolderMesh, = makeOutputFolder(inputs.outputFolder, cfg, Prefix='XYZRestpose_')
    print('outFolderForExperiment: ', outFolderForExperiment)
    rendererSynth = Renderer(device, cfg)

    # initial image
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize)
    meshes = join_meshes_as_batch([smplshMesh for i in range(cfg.batchSize)])
    images = renderImages(cams, rendererSynth, meshes, )
    visualize2DSilhouetteResults(images, backGroundImages = crops_out, outImgFile=join(outFolderForExperiment, 'Fit0_Initial.png'))

    # initial diff image
    outFolderDiffImage = join(outFolderForExperiment, 'DiffImg')
    os.makedirs(outFolderDiffImage, exist_ok=True)
    diffImages, loss = getDiffImages(images, crops_out, cams, cfg)
    print("Initial loss:", loss)
    visualize2DResults(diffImages, outImgFile=join(outFolderDiffImage, 'Fit0_Initial.png'))

    poses = []
    losses = []
    optimizer = torch.optim.Adam([trans, pose, betas, xyzShift], lr=cfg.learningRate)
    logFile = join(outFolderForExperiment, 'Logs.txt')
    logger = Logger.configLogger(logFile)

    fitParamFolder = join(outFolderForExperiment, 'FitParam')
    os.makedirs(fitParamFolder, exist_ok=True)

    imagesBatchRefs = []
    for iCam in range(len(cams)):
        imagesBatchRef = crops_out[iCam * cfg.batchSize:iCam * cfg.batchSize + cfg.batchSize, ..., 0]
        imagesBatchRef = torch.tensor(imagesBatchRef, dtype=torch.float32, device=device, requires_grad=False)
        imagesBatchRefs.append(imagesBatchRef)

    loop = tqdm(range(cfg.numIterations))

    for i in loop:
        optimizer.zero_grad()

        lossVal = 0
        for iCam in range(len(cams)):
            verts = smplsh(betas, pose, trans).type(torch.float32)
            #         modifiedVerts = verts + xyzShift
            mesh = Meshes(verts=[verts], faces=[smplsh.faces.to(device)], )
            meshes = join_meshes_as_batch([mesh for i in range(cfg.batchSize)])
            imagesBatchRef = imagesBatchRefs[iCam]

            images = rendererSynth.renderer(meshes, cameras=cams[iCam])
            loss = 0
            for iIm in range(cfg.batchSize):
                loss += (1 - torch.norm(imagesBatchRef[iIm, ...] * images[iIm, ..., 3], p=1) / torch.norm(
                    imagesBatchRef[iIm, ...] + images[iIm, ..., 3] - imagesBatchRef[iIm, ...] * images[iIm, ..., 3],
                    p=1)) / cfg.numCams

            loss.backward()
            lossVal += loss.item()

        #     modifiedVerts = verts + xyzShift
        verts = smplsh(betas, pose, trans).type(torch.float32)
        mesh = Meshes(
            verts=[verts], faces=[smplsh.faces.to(device)], )

        #     loss = cfg.lpSmootherW * mesh_laplacian_smoothing(mesh) + cfg.normalSmootherW * mesh_normal_consistency(mesh)
        loss = cfg.normalSmootherW * mesh_normal_consistency(mesh)
        normalSmootherVal = loss.item()
        loss = loss + cfg.lpSmootherW * xyzShift[:, 0:1].transpose(0, 1) @ LNP @ xyzShift[:, 0:1]
        loss = loss + cfg.lpSmootherW * xyzShift[:, 1:2].transpose(0, 1) @ LNP @ xyzShift[:, 1:2]
        loss = loss + cfg.lpSmootherW * xyzShift[:, 2:3].transpose(0, 1) @ LNP @ xyzShift[:, 2:3]
        lpSmootherVal = loss.item() - normalSmootherVal

        loss.backward()
        lossVal += loss.item()
        # to corners loss
        verts = smplsh(betas, pose, trans).type(torch.float32)
        loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        loss.backward()
        #     lossVal += loss.item()
        toSparseCloudLoss = loss.item()

        # fixing loss
        loss = torch.sum(xyzShift[indicesToFix, :] ** 2)
        loss.backward()
        #     lossVal += loss.item()
        # recordData
        losses.append(lossVal)

        optimizer.step()
        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'Fitting loss %.6f, normal regularizer loss %.6f, Laplacian regularizer loss %.6f, toSparseCloudLoss %.6f, MemUsed:%.2f' \
                  % (lossVal, normalSmootherVal, lpSmootherVal, toSparseCloudLoss, memAllocated)

        loop.set_description(infoStr)
        logger.info(infoStr)

        # Save outputs to create a GIF.
        if (i+1) % cfg.plotStep == 0 and i:
            showCudaMemUsage(device)

            torch.cuda.empty_cache()
            plt.close('all')

            with torch.no_grad():
                verts = smplsh(betas, pose, trans).type(torch.float32)
                #         modifiedVerts = verts + xyzShift
                mesh = Meshes(verts=[verts], faces=[smplsh.faces.to(device)], )
                meshes = join_meshes_as_batch([mesh for i in range(cfg.batchSize)])
                images = renderImages(cams, rendererSynth, meshes, )
                visualize2DSilhouetteResults(images, backGroundImages=crops_out,
                                             outImgFile=join(outFolderForExperiment, 'Fit' + str(i).zfill(5) + '.png'))
                # initial diff image
                diffImages, loss = getDiffImages(images, crops_out, cams, cfg)
                visualize2DResults(diffImages, outImgFile=join(outFolderDiffImage, 'Fit' + str(i).zfill(5) + '.png'))

            outParamFile = join(fitParamFolder, 'Param_' + str(i).zfill(5) + '.npz')
            np.savez(outParamFile, trans=trans.cpu().detach().numpy(), pose=pose.cpu().detach().numpy(),
                     beta=betas.cpu().detach().numpy(), personalShape=xyzShift.cpu().detach().numpy())

            saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

if __name__ == '__main__':
    inputs = InputBundle()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cfgPoseFitting = RenderingCfg()

    cfgPoseFitting.sigma = 1e-7
    cfgPoseFitting.blurRange = 1e-7
    # cfgPoseFitting.plotStep = 20
    cfgPoseFitting.plotStep = 20
    cfgPoseFitting.numCams = 16
    # low learning rate for pose optimization
    cfgPoseFitting.learningRate = 1e-3
    cfgPoseFitting.batchSize = 4
    # cfgPoseFitting.faces_per_pixel = 6 # for testing
    cfgPoseFitting.faces_per_pixel = 15 # for debugging
    # cfgPoseFitting.imgSize = 2160
    cfgPoseFitting.imgSize = 1080
    cfgPoseFitting.terminateLoss = 0.1
    cfgPoseFitting.lpSmootherW = 0.000001
    # cfgPoseFitting.normalSmootherW = 0.1
    cfgPoseFitting.normalSmootherW = 0.0
    cfgPoseFitting.numIterations = 500
    # cfgPoseFitting.numIterations = 20
    cfgPoseFitting.kpFixingWeight = 0

    cfgPerVert = RenderingCfg()
    cfgPerVert.sigma = 1e-7
    cfgPerVert.blurRange = 1e-7
    cfgPerVert.plotStep = 20
    # cfgPerVert.plotStep = 5
    cfgPerVert.numCams = 16
    cfgPerVert.learningRate = 0.1
    # cfgPerVert.batchSize = 2
    cfgPerVert.batchSize = 4
    cfgPerVert.faces_per_pixel = 6
    # cfgPerVert.faces_per_pixel = 15

    # cfgPerVert.imgSize = 2160
    cfgPerVert.imgSize = 1080
    device = torch.device("cuda:0")
    cfgPerVert.terminateLoss = 0.1
    cfgPerVert.lpSmootherW = 0.000001
    cfgPerVert.normalSmootherW = 0.0
    cfgPerVert.numIterations = 500
    # cfgPerVert.numIterations = 20

    fileNames = ['03067', '04735', '06550']

    for frameName in fileNames:
        inputs.imageFolder = join(r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting', frameName, 'silhouettes')
        inputs.sparsePointCloudFile =join(r'Z:\shareZ\2020_05_21_AC_FramesDataToFitTo\Copied', frameName, 'A000' + frameName + '.obj')
        # inputs.outputFolder = join(r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\Output', frameName)
        inputs.outputFolder = join(r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Output', frameName)

        inputs.initialFittingParamFile = join(r'Z:\shareZ\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams', frameName+'.npz')
        inputs.KeypointsFile = join(r'Z:\shareZ\2020_06_14_FitToMultipleCams\KepPoints', frameName +'.obj')

        inputsPose = copy(inputs)
        inputsPose.outputFolder = join(inputs.outputFolder, 'SilhouettePose')
        # toSilhouettePoseFitting(inputsPose, cfgPoseFitting)
        poseFittingParamFolder, _ = makeOutputFolder(inputsPose.outputFolder, cfgPoseFitting, Prefix='PoseFitting_')
        paramFiles = glob.glob(join(poseFittingParamFolder, 'FitParam', '*.npz'))
        paramFiles.sort()
        finalPoseFile = paramFiles[-1]

        inputsPerVertFitting = copy(inputs)
        inputsPerVertFitting.imageFolder = join(inputs.imageFolder, 'Undist')
        inputsPerVertFitting.outputFolder = join(inputs.outputFolder, 'SilhouettePerVert')
        inputsPerVertFitting.initialFittingParamFile = finalPoseFile
        # toSilhouettePerVertexFitting(inputsPerVertFitting, cfgPerVert)
        perVertFittingFolder, _ = makeOutputFolder(inputsPerVertFitting.outputFolder, cfgPerVert, Prefix='XYZRestpose_')

        # copy final data
        outFolderFinalData = join(inputs.finalOutputFolder, frameName)
        os.makedirs(outFolderFinalData, exist_ok=True)
        imageFiles = glob.glob(join(perVertFittingFolder, '*.png'))
        imageFiles.sort()
        finalImgFile = imageFiles[-2]
        shutil.copy(finalImgFile, join(outFolderFinalData, os.path.basename(finalImgFile)))

        fitParamFiles = glob.glob(join(perVertFittingFolder, 'FitParam', '*.npz'))
        fitParamFiles.sort()
        finalParamFile = fitParamFiles[-1]
        shutil.copy(finalParamFile, join(outFolderFinalData, 'FitParam_' + os.path.basename(finalParamFile)))

        meshFiles = glob.glob(join(perVertFittingFolder, 'mesh', '*.vtk'))
        meshFiles.sort()
        finalMesh = meshFiles[-1]
        shutil.copy(finalMesh, join(outFolderFinalData, 'PerVertex_' + os.path.basename(finalMesh)))

        outIntepolatedMesh = join(outFolderFinalData, 'InterpolatedMesh.ply')
        interpolateWithSparsePointCloudSoftly(finalMesh, inputs.sparsePointCloudFile, outIntepolatedMesh,
            '06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json', laplacianMatFile='SmplshRestposeLapMat.npy')






