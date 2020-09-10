from M05_Visualization import *
import tqdm
import shutil

class InputBundle():
    def __init__(s):
        s.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'
        s.smplshExampleMeshFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\SMPL_Socks\SMPLSH\SMPLSH.obj'
        s.cleanPlateFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
        s.texturedMesh = r"..\Data\TextureMap2Color\SMPLWithSocks_tri.obj"
        s.compressedStorage = True

        s.inFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
        s.finalMeshFolder = None
        s.cleanPlateFolder = None
        s.outFolder = None
        s.imgParentFolder = None

def makeComparison(frameNames, inputs, cfg, imgExt='png'):
    renderConsecutiveFrames(inputs.inFolder, inputs.cleanPlateFolder, inputs.texturedMesh, inputs.camParamF, inputs.outFolder, frameNames=frameNames, cfg=cfg)

    # copy reference images
    if frameNames is None:
        imageFolders = glob.glob(join(inputs.imgParentFolder, '*'))
    else:
        imageFolders = [join(inputs.imgParentFolder, frameName) for frameName in frameNames]

    camNames = ['A', 'B', 'C', 'D', 'E', "F", 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
    referenceOutFolders = []
    renderedImgFolders = []
    renderedImgFiles = []
    flipComparisonFolders = []
    sideBySideComparisonFolders = []

    for i in range(len(camNames)):
        camName = camNames[i]
        refOutFolder = join(inputs.outFolder, camName, 'Reference')
        os.makedirs(refOutFolder, exist_ok=True)
        referenceOutFolders.append(refOutFolder)

        renderedOutFolder = join(inputs.outFolder, camName, 'Rendered')
        renderedImgFolders.append(renderedOutFolder)
        if frameNames is None:
            renderedImgFiles.append(glob.glob(join(renderedOutFolder, '*.png')))
        else:
            renderedImgFiles.append([join(renderedOutFolder, frameName + '.png') for frameName in frameNames])

        flipComparisonFolder = join(inputs.outFolder, camName, 'FlipComparison')
        os.makedirs(flipComparisonFolder, exist_ok=True)
        flipComparisonFolders.append(flipComparisonFolder)

        sideBySideComparisonFolder = join(inputs.outFolder, camName, 'SideBySideComparison')
        os.makedirs(sideBySideComparisonFolder, exist_ok=True)
        sideBySideComparisonFolders.append(sideBySideComparisonFolder)

    for iFrame, imgFolder in tqdm.tqdm(enumerate(imageFolders), desc='Copy reference images.'):
        imgFiles = glob.glob(join(imgFolder, '*.' + imgExt))
        imgFiles.sort()

        # for iCam, imgF in enumerate(imgFiles):
        #     shutil.copy(imgF, join(referenceOutFolders[iCam], os.path.basename(imgF)))
        # image_refs_out, crops_out = load_images(imgFolder, camParamF=inputs.camParamF, UndistImgs=True,
        #                                         cropSize=cfg.imgSize, imgExt='pgm', writeUndistorted=False, normalize=False, flipImg=False)
        image_refs_out, crops_out = load_images(imgFolder, camParamF=inputs.camParamF, UndistImgs=False,
                                                cropSize=cfg.imgSize, imgExt=imgExt, writeUndistorted=False,
                                                normalize=False, flipImg=False, cvtToRGB=False)

        for iCam, imgF in enumerate(imgFiles):
            cv2.imwrite(join(referenceOutFolders[iCam], os.path.basename(imgF) + '.png'),
                        crops_out[iCam].astype(np.uint8))

            # make flip comparison
            shutil.copy(renderedImgFiles[iCam][iFrame],
                        join(flipComparisonFolders[iCam], Path(imgF).stem + '.0Rendered.png'))
            cv2.imwrite(join(flipComparisonFolders[iCam], os.path.basename(imgF) + '.png'),
                        crops_out[iCam].astype(np.uint8))

            # make side by side comparison
            collageImage = np.concatenate(
                [cv2.imread(renderedImgFiles[iCam][iFrame]), crops_out[iCam].astype(np.uint8)], axis=1)
            # cv2.imshow('sideBySideComparison', collageImage)
            # cv2.waitKey()
            cv2.imwrite(join(sideBySideComparisonFolders[iCam], os.path.basename(imgF) + '.png'), collageImage)

if __name__ == '__main__':
    inputs = InputBundle()
    inputs.inFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Final\Mesh'
    inputs.cleanPlateFolder =  r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    # inputs.outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Comparison\HandOnGround'
    inputs.outFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\HandOnGround'
    inputs.imgParentFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Preprocessed'
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
    cfg.ambientLvl = 0.8

    frameNames = [
        str(i).zfill(5) for i in range(8564, 8708


                                       )
        # str(i).zfill(5) for i in range(8820, 8820 + 3)

    ]

    makeComparison(frameNames, inputs, cfg, imgExt='png')