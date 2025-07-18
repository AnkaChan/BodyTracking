from Config import *
from Utility import *
from copy import copy
from tqdm import tqdm
import shutil
from S01_RegisterSparsePointCloud import getInterpoMatSubdivision
from S13_GetPersonalShapeFromInterpolation import getPersonalShape
from S05_InterpolateWithSparsePointCloud import interpolateSubdivMesh

class InputBundle:
    def __init__(s, datasetName=r'Lada_12/12/2019'):
        if datasetName == r'Lada_12/12/2019':
            # same over all frames
            s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
            s.smplshExampleMeshFile = r'..\SMPL_reimp\SMPLSH.obj'
            s.toSparsePCMat = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolationMatrix.npy'
            s.smplshRegressorMatFile = r'C:\Code\MyRepo\ChbCapture\08_CNNs\Openpose\SMPLSHAlignToAdamWithHeadNoFemurHead\smplshRegressorNoFlatten.npy'
            s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'

            s.handIndicesFile = r'HandIndices.json'
            s.HeadIndicesFile = r'HeadIndices.json'
            s.personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'
            s.texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"

            # frame specific inputs
            s.imageFolder = r'F:\WorkingCopy2\2020_06_04_SilhouetteExtraction\03067\silhouettes'
            s.KeypointsFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\KepPoints\03067.obj'
            s.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\03067\A00003067.obj'

            s.compressedStorage = True
            s.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03067.npz'
            s.outputFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\03067'
            # copy all the final result to this folder
            s.finalOutputFolder = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\Final'
        elif  datasetName == r'Katey_01/01/2020':
            # same over all frames
            s.camParamF = r'F:\WorkingCopy2\2020_01_01_KateyCapture\CameraParameters3_k6p2\cam_params.json'
            s.smplshExampleMeshFile = r'..\SMPL_reimp\SMPLSH.obj'
            s.toSparsePCMat = r''
            s.smplshRegressorMatFile = r''
            s.smplshData = r'..\SMPL_reimp\SmplshModel_f_noBun.npz'
            s.skelDataFile = r'..\Data\KateyBodyModel\InitialRegistration\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'

            s.handIndicesFile = r'HandIndices.json'
            s.HeadIndicesFile = r'HeadIndices.json'
            s.personalShapeFile = r''
            s.texturedMesh = r"..\Data\TextureMap\SMPLWithSocks.obj"

            # frame specific inputs
            s.imageFolder = None
            s.KeypointsFile = None
            s.sparsePointCloudFile = None

            s.compressedStorage = True
            s.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\03067.npz'
            s.outputFolder = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\Output\03067'
            # copy all the final result to this folder
            s.finalOutputFolder = None

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
        + 'LPW_' + str(cfg.lpSmootherW) + 'HHW_' + str(cfg.vertexFixingWeight) + '_It' + str(cfg.numIterations)

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
                iCam = numCols * iRow + iCol
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

def faceKpLossTorch(smplshVerts, keypoints):
    corrs =np.array( [
        [75, 3161],  # middle chin
        [115, 3510],  # right mouth corner
        [121, 69],  # left mouth corner
        [74, 3544],
        [76, 285],

    ])

    return torch.mean(((smplshVerts[corrs[:,1], :]) - keypoints[corrs[:,0]])**2)

def toSilhouettePoseInitalFitting(inputs, cfg, device, undistortSilhouettes=False):
    OPHeadKeypoints = [0, 15, 16, 17, 18]
    smplshExampleMesh = pv.PolyData(inputs.smplshExampleMeshFile)

    # The head joint regressor and keypoint data
    Keypoints = pv.PolyData(inputs.KeypointsFile)
    headKps = torch.tensor(Keypoints.points[OPHeadKeypoints, :], dtype=torch.float32, device=device,
                           requires_grad=False)

    jointConverter = VertexToOpJointsConverter()

    # Read fitting parameter
    if inputs.compressedStorage:
        transInit, poseInit, betaInit = loadCompressedFittingParam(inputs.initialFittingParamFile)
    else:
        transInit = np.load(inputs.initialFittingParamTranslationFile)
        poseInit = np.load(inputs.initialFittingParamPoseFile)
        betaInit = np.load(inputs.initialFittingParamBetasFile)
    # Make fitting parameter tensors
    pose = torch.tensor(poseInit, dtype=torch.float64, requires_grad=True, device=device)
    betas = torch.tensor(betaInit, dtype=torch.float64, requires_grad=False, device=device)
    trans = torch.tensor(transInit, dtype=torch.float64,
                         requires_grad=True, device=device)

    # Build up smplsh model
    smplsh = smplsh_torch.SMPLModel(device, inputs.smplshData, personalShape=None)
    verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
    smplshMesh = Meshes([verts], [smplsh.faces.to(device)])

    # Build up the sparse point cloud constraint
    # interpoMat = np.load(inputs.toSparsePCMat)
    # registeredCornerIds = np.where(np.any(interpoMat, axis=1))[0]
    # print("Number of registered corners:", registeredCornerIds.shape)

    # sparsePC = pv.PolyData(inputs.sparsePointCloudFile)
    # sparsePC = np.array(sparsePC.points)
    # constraintIds = np.where(sparsePC[:, 2] > 0)[0]
    # constraintIds = np.intersect1d(registeredCornerIds, constraintIds)
    # print("Number of constraint corners:", constraintIds.shape)
    #
    # interpoMat = interpoMat[constraintIds, :]
    # sparsePC = sparsePC[constraintIds, :]
    #
    # # initial to sparse point cloud dis
    # sparsePC = torch.tensor(sparsePC, dtype=torch.float32, requires_grad=False, device=device)
    # interpoMat = torch.tensor(interpoMat, dtype=torch.float32, requires_grad=False, device=device)

    # load camera and distort image
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape)
    cams = init_camera_batches(cams_torch, device)

    image_refs_out, crops_out = load_images(inputs.imageFolder, cropSize=cfg.imgSize, UndistImgs=undistortSilhouettes, camParamF=inputs.camParamF)
    outFolderForExperiment, outFolderMesh, = makeOutputFolder(inputs.outputFolder, cfg, Prefix='PoseFitting_')
    print('outFolderForExperiment:', outFolderForExperiment)

    # build renderer
    rendererSynth = Renderer(device, cfg)

    # initial image
    images = renderImages(cams, rendererSynth, smplshMesh, )
    visualize2DSilhouetteResults(images, backGroundImages = crops_out, outImgFile=join(outFolderForExperiment, 'Fit0_Initial.png'))
    saveVTK(join(outFolderMesh, 'Fit0_Initial.ply'), verts.cpu().detach().numpy(),
            smplshExampleMesh)

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
            # Intersection over union loss
            loss = 1 - torch.norm(refImg * images[..., 3], p=1) / torch.norm(
                refImg + images[..., 3] - refImg * images[..., 3], p=1)

            loss.backward()
            lossVal += loss.item()

        # joint regularizer
        loss = cfg.jointRegularizerWeight * torch.sum((pose ** 2))
        loss.backward()

        # # to corners loss
        # verts = smplsh(betas, pose, trans).type(torch.float32) * 1000
        # loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        # loss.backward()
        # #     lossVal += loss.item()
        # toSparseCloudLoss = loss.item()

        # recordData
        verts, jointsDeformed = smplsh(betas, pose, trans, returnDeformedJoints=True)
        verts = verts.type(torch.float32) * 1000
        jointsDeformed = jointsDeformed.type(torch.float32) * 1000
        #     headJoints = smplshRegressorMatHead @ verts
        smplshOpJoints = jointConverter(verts[None, ...], jointsDeformed[None, ...])
        headJoints = \
        torch.index_select(smplshOpJoints, 1, torch.tensor(OPHeadKeypoints, dtype=torch.long, device=device))[0, ...]

        loss = cfg.kpFixingWeight * torch.sum((headJoints - headKps) ** 2)
        loss.backward()
        headKpFixingLoss = loss.item()

        losses.append(lossVal)

        if i:
            optimizer.step()

        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'image loss %.6f, headKpFixingLoss %.4f, MemUsed:%.2f' \
                  % (lossVal, headKpFixingLoss, memAllocated)

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
                     beta=betas.cpu().detach().numpy(), )
            visualize2DSilhouetteResults(renderedImages, backGroundImages=crops_out, outImgFile=outImgFile,
                                         sizeInInches=5)

            saveVTK(join(outFolderMesh, 'Fit' + str(i).zfill(5) + '.ply'), verts.cpu().detach().numpy(),
                    smplshExampleMesh)

def toSilhouettePerVertexInitialFitting(inputs, cfg, device):
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
    initialPersonalShape = np.full((nVerts,3), 0)
    xyzShift = torch.tensor(initialPersonalShape, dtype=torch.float32, requires_grad=True, device=device)

    # load Images
    image_refs_out, crops_out = load_images(inputs.imageFolder, cropSize=cfg.imgSize)
    crops_out = np.stack(crops_out, axis=0)

    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape)

    # Build up the sparse point cloud constraint
    # interpoMat = np.load(inputs.toSparsePCMat)
    #
    # registeredCornerIds = np.where(np.any(interpoMat, axis=1))[0]
    # print("Number of registered corners:", registeredCornerIds.shape)

    # sparsePC = pv.PolyData(inputs.sparsePointCloudFile)
    # sparsePC = np.array(sparsePC.points)
    #
    # constraintIds = np.where(sparsePC[:, 2] > 0)[0]
    # constraintIds = np.intersect1d(registeredCornerIds, constraintIds)
    # print("Number of constraint corners:", constraintIds.shape)
    #
    # interpoMat = interpoMat[constraintIds, :]
    # sparsePC = sparsePC[constraintIds, :]
    # # initial to sparse point cloud dis
    #
    # sparsePC = torch.tensor(sparsePC, dtype=torch.float32, requires_grad=False, device=device)
    # interpoMat = torch.tensor(interpoMat, dtype=torch.float32, requires_grad=False, device=device)

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
    saveVTK(join(outFolderMesh, 'Fit0_Initial.ply'), verts.cpu().detach().numpy(),
            smplshExampleMesh)

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
        # # to corners loss
        # verts = smplsh(betas, pose, trans).type(torch.float32)
        # loss = cfg.toSparseCornersFixingWeight * torch.sum((sparsePC - interpoMat @ verts) ** 2)
        # loss.backward()
        # #     lossVal += loss.item()
        # toSparseCloudLoss = loss.item()

        # fixing loss
        loss = cfg.vertexFixingWeight * torch.sum(xyzShift[indicesToFix, :] ** 2)
        loss.backward()
        hhFixingLoss = loss.item()
        # recordData
        losses.append(lossVal)

        optimizer.step()
        memStats = torch.cuda.memory_stats(device=device)
        memAllocated = memStats['active_bytes.all.current'] / 1000000
        torch.cuda.empty_cache()

        infoStr = 'Fitting loss %.6f, normal regularizer loss %.6f, Laplacian regularizer loss %.6f, hhFixingLoss %.6f, MemUsed:%.2f' \
                  % (lossVal, normalSmootherVal, lpSmootherVal, hhFixingLoss, memAllocated)

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

def getRestposeLapMat(subdivMesh, restposeMesh, outSubdivCorr, registrationTId, registrationBarys, outRestposeLapMat):
    restposeMesh = pv.PolyData(restposeMesh)
    subdivMesh = pv.PolyData(subdivMesh)

    subdivMesh.points[restposeMesh.points.shape[0], :] = restposeMesh.points

    outSubdivCorr = 0

    pass

if __name__ == '__main__':
    ### This is the firsting fitting to silhouette, before we have it register to sparse point cloud
    # before we have the texture


    inputs = InputBundle('Katey_01/01/2020')
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cfgPoseFitting = RenderingCfg()

    cfgPoseFitting.sigma = 1e-7
    cfgPoseFitting.blurRange = 1e-7
    # cfgPoseFitting.plotStep = 20
    cfgPoseFitting.plotStep = 50
    cfgPoseFitting.numCams = 16
    # low learning rate for pose optimization
    cfgPoseFitting.learningRate = 1e-4
    cfgPoseFitting.batchSize = 16
    # cfgPoseFitting.faces_per_pixel = 6 # for testing
    cfgPoseFitting.faces_per_pixel = 6 # for debugging
    # cfgPoseFitting.imgSize = 2160
    # cfgPoseFitting.imgSize = 1080
    cfgPoseFitting.imgSize = 540
    cfgPoseFitting.terminateLoss = 0.1
    cfgPoseFitting.lpSmootherW = 0.000001
    # cfgPoseFitting.normalSmootherW = 0.1
    cfgPoseFitting.normalSmootherW = 0.0
    # cfgPoseFitting.numIterations = 300
    cfgPoseFitting.numIterations = 500
    # cfgPoseFitting.numIterations = 20
    cfgPoseFitting.kpFixingWeight = 0.005
    cfgPoseFitting.bin_size = None

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
    # cfgPerVert.lpSmootherW = 0.000001
    cfgPerVert.lpSmootherW = 0.0000001
    cfgPerVert.normalSmootherW = 0.0
    # cfgPerVert.numIterations = 500
    cfgPerVert.numIterations = 300
    cfgPerVert.optimizePose = False
    # cfgPerVert.numIterations = 20
    cfgPerVert.bin_size = 256

    # For Lada
    # frameName = '3052'
    # undistortSilhouette = False
    #
    # inputs.imageFolder = r'F:\WorkingCopy2\2020_06_04_SilhouetteExtraction\3052\Silhouette'
    # # inputs.outputFolder = join(r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\Output', frameName)
    # inputs.outputFolder = join(r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting', frameName)
    #
    # inputs.compressedStorage = False
    # inputs.initialFittingParamPoseFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\NewInitialFitting\InitialRegistration\OptimizedPoses_ICPTriangle.npy'
    # inputs.initialFittingParamBetasFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\NewInitialFitting\InitialRegistration\OptimizedBetas_ICPTriangle.npy'
    # inputs.initialFittingParamTranslationFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\NewInitialFitting\InitialRegistration\OptimizedTranslation_ICPTriangle.npy'
    #
    # inputs.KeypointsFile = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\KepPoints\00352.obj'

    # For Katey
    frameName = '18411'
    undistortSilhouette = True

    inputs.sparsePointCloudFile = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TPose\Deformed\A00018411.obj'
    inputs.imageFolder = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\Silhouettes\Sihouettes_NoGlassese\18411'
    # inputs.outputFolder = join(r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\Output', frameName)
    inputs.outputFolder = join(r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\InitialSilhouetteFitting_NoGlassese', frameName)
    inputs.finalOutputFolder = join(r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\InitialSilhouetteFitting_NoGlassese', 'FinalSubdiv')
    inputs.compressedStorage = False
    inputs.initialFittingParamPoseFile = r'..\Data\KateyBodyModel\InitialRegistration\OptimizedPoses_ICPTriangle.npy'
    inputs.initialFittingParamBetasFile = r'..\Data\KateyBodyModel\InitialRegistration\OptimizedBetas_ICPTriangle.npy'
    inputs.initialFittingParamTranslationFile = r'..\Data\KateyBodyModel\InitialRegistration\OptimizedTranslation_ICPTriangle.npy'
    inputs.outFittingParamFileWithPS =  r'..\Data\KateyBodyModel\FitParamsWithPersonalShape.npz'
    inRestposeSMPLSHMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\BuildSmplsh_Female\InterpolateFemaleShape\SMPLWithSocks_tri_Aligned_female_NoBun.obj'

    inputs.toSparsePCMat = '..\Data\KateyBodyModel\InterpolationMatrix.npy'

    inputs.KeypointsFile = r'F:\WorkingCopy2\2020_08_27_KateyBodyModel\TPose\Keypoints\18411.obj'

    inputsPose = copy(inputs)
    inputsPose.outputFolder = join(inputs.outputFolder, 'SilhouettePose')
    # toSilhouettePoseInitalFitting(inputsPose, cfgPoseFitting, device, undistortSilhouettes=undistortSilhouette)
    poseFittingParamFolder, _ = makeOutputFolder(inputsPose.outputFolder, cfgPoseFitting, Prefix='PoseFitting_')
    paramFiles = glob.glob(join(poseFittingParamFolder, 'FitParam', '*.npz'))
    paramFiles.sort()
    finalPoseFile = paramFiles[-1]

    inputsPerVertFitting = copy(inputs)
    if undistortSilhouette:
        inputsPerVertFitting.imageFolder = join(inputs.imageFolder, 'Undist')
    else:
        inputsPerVertFitting.imageFolder = inputs.imageFolder

    inputsPerVertFitting.outputFolder = join(inputs.outputFolder, 'SilhouettePerVert')
    inputsPerVertFitting.compressedStorage = True
    inputsPerVertFitting.initialFittingParamFile = finalPoseFile
    # toSilhouettePerVertexInitialFitting(inputsPerVertFitting, cfgPerVert, device)
    perVertFittingFolder, _ = makeOutputFolder(inputsPerVertFitting.outputFolder,
                                               cfgPerVert, Prefix='XYZRestpose_')

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

    meshFiles = glob.glob(join(perVertFittingFolder, 'mesh', '*.ply'))
    meshFiles.sort()
    finalMesh = meshFiles[-2]
    shutil.copy(finalMesh, join(outFolderFinalData, 'PerVertex_' + os.path.basename(finalMesh)))

    outIntepolatedMesh = join(outFolderFinalData, 'InterpolatedMesh.ply')
    outSubdivMesh = join(outFolderFinalData, 'Subdivied.ply')
    outSubdivCorr = join(outFolderFinalData, 'SubdivCorrs.txt')

    # getInterpoMat(finalMesh, inputs.sparsePointCloudFile, inputs.toSparsePCMat, inputs.skelDataFile, minValInterpoMat=0.1)
    # getInterpoMatSubdivision(finalMesh, inputs.sparsePointCloudFile, inputs.toSparsePCMat, inputs.skelDataFile,
    #                          outSubdivCorr, outSubdivMesh, minValInterpoMat=0.3)


    registrationTIdFile = join(outFolderFinalData, 'RegistrationTId.npy')
    registrationBarysFile = join(outFolderFinalData, 'RegistrationBarys.npy')
    outRestposeLapMat = join(outFolderFinalData, 'RestposeLapMat.npy')

    # getRestposeLapMat(outSubdivMesh, outSubdivCorr, registrationTIdFile, registrationBarysFile, outRestposeLapMat)

    interpolateSubdivMesh(outSubdivMesh, inputs.sparsePointCloudFile, outIntepolatedMesh, inputs.skelDataFile, outSubdivCorr,
             laplacianMatFile=None, biLaplacian=False, interpolateDisplacement=True)
    #'SmplshRestposeLapMat.npy'

    getPersonalShape(outIntepolatedMesh, finalParamFile, inputs.outFittingParamFileWithPS, inputs.smplshData)

    # should do it in the restpose



