from Config import *
from S02_TexturedFittingForSelectedFrames import InputBundle
from Utility_Rendering import *

if __name__ == '__main__':
    inputs = InputBundle()

    inputs.imageFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\Preprocessed\03067'
    inputs.sparsePointCloudFile = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step8_Overlap0\Deformed\A00003067.obj'
    inputs.cleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    inputs.compressedStorage = True
    # inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\FitToSparseCloud\FittingParams\06950.npz'
    inputs.initialFittingParamFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr\ToSparseFittingParams.npz'
    inDeformedMeshFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr\ToSparseFitFinalMesh.obj'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\TextureCompletionFitting\03067_Old2DKpErr'
    inputs.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
    inputs.outputFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\TestMultiResolution\03067'

    # inputs.texturedMesh = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj'
    inputs.texturedMesh = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\TextureFaceCulling\Body\Data\03067.obj'
    # inputs.texturedMesh = r'C:\Code\MyRepo\07_Graphics\pytorch3d_0.2.0\docs\tutorials\data\cow_mesh\cow.obj'
    # inputs.texturedMesh = r'C:\Code\MyRepo\07_Graphics\pytorch3d_0.2.0\docs\tutorials\data\cow_mesh\cow_inverseOrientation.obj'
    # outFolderForExperiment = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\TextureFaceCulling\cow'
    outFolderForExperiment = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\TextureFaceCulling\Body'
    os.makedirs(outFolderForExperiment, exist_ok=True)

    cfg = RenderingCfg()
    cfg.sigma = 1e-7
    cfg.blurRange = 1e-7

    cfg.sigma = 1e-8
    cfg.blurRange = 1e-8
    cfg.gamma = 1e-8

    cfg.numIterations = 2000
    cfg.plotStep = 1000
    cfg.numCams = 16
    # low learning rate for pose optimization
    # cfg.learningRate = 1e-4
    cfg.learningRate = 1e-3

    cfg.faces_per_pixel = 5  # for debugging
    # cfg.faces_per_pixel = 1  # for debugging
    # cfg.faces_per_pixel = 1  # for debugging
    # cfg.imgSize = 2160
    # cfg.imgSize = 1080
    # cfg.batchSize = 2

    cfg.imgSize = 1080
    # cfg.imgSize = 540
    # cfg.imgSize = 256
    cfg.batchSize = 1

    cfg.terminateLoss = 0.1
    cfg.lpSmootherW = 1e-10
    # cfg.normalSmootherW = 0.1
    cfg.normalSmootherW = 0.0
    # cfg.numIterations = 20
    cfg.useKeypoints = False
    cfg.kpFixingWeight = 0
    cfg.bodyJointOnly = True
    cfg.jointRegularizerWeight = 1e-5
    # cfg.bin_size = 256
    # cfg.bin_size = None
    cfg.bin_size = 0
    cfg.inputImgExt = 'png'
    cfg.terminateStep = 1e-6
    # cfg.cull_backfaces = True
    cfg.cull_backfaces = False
    cfg.makeOutputSubfolder = True

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    mesh = load_objs_as_meshes([inputs.texturedMesh], device=device)
    mesh = mesh.update_padded(mesh.verts_padded() / 1000)

    # set up light
    xyz = torch.from_numpy(np.float32([0, 0, 2]))[None]
    diffuse = 0.5
    ambient = cfg.ambientLvl
    specular = 0.5
    s = specular * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    d = diffuse * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    a = ambient * torch.from_numpy(np.ones((1, 3)).astype(np.float32)).to(device)
    light = PointLights(device=device, location=xyz, specular_color=s, ambient_color=a, diffuse_color=d)
    rendererSynth = RendererWithTexture(device, lights=light, cfg=cfg)

    # load cameras
    actual_img_shape = (2160, 4000)
    cam_params, cams_torch = load_cameras(inputs.camParamF, device, actual_img_shape, unitM=True)
    cams = init_camera_batches(cams_torch, device, batchSize=cfg.batchSize)

    # cams[0].transform_points(mesh.verts_padded())
    tranformsW2V = cams[0].get_world_to_view_transform()
    pointsInView = tranformsW2V.transform_points(mesh.verts_padded())

    bodyMesh = pv.PolyData(inputs.smplshExampleMeshFile)
    bodyMesh.points = np.array(pointsInView.cpu().numpy()[0,...])
    bodyMesh.save(join(outFolderForExperiment, 'BodyInView.ply'))


    # load backgrounds
    cp_out, cp_crop_out = load_images(inputs.cleanPlateFolder, cropSize=cfg.imgSize, UndistImgs=False, camParamF=inputs.camParamF,
                                      imgExt='png')
    backgrounds = []
    for iBatch in range(len(cams)):
        bg = cp_crop_out[iBatch*cfg.batchSize:(iBatch+1)*cfg.batchSize:]
        bg = torch.tensor(bg, dtype=torch.float32, device=device, requires_grad=False)
        backgrounds.append(bg)

    # render images
    meshes = join_meshes_as_batch([mesh for i in range(cfg.batchSize)])
    images = renderImagesWithBackground(cams, rendererSynth, meshes, backgrounds)

    descName = ''.join(['_Size', str(cfg.imgSize), '_Fpp', str(cfg.faces_per_pixel), '_Sig', str(cfg.sigma), '_BFCulling', str(cfg.cull_backfaces), '_gamma' + str(cfg.gamma)])

    visualize2DResults(images, outImgFile=join(outFolderForExperiment, 'Results' +descName+ '.png'), withAlpha=False,
                       sizeInInches=5)



