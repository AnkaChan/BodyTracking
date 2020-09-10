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

if __name__ == '__main__':
    inputs = InputBundle()
    inputs.inFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\ToSparse'
    inputs.cleanPlateFolder =  r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    # inputs.outFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Comparison\HandOnGround'
    inputs.outFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\Evaluation'
    inputs.imgParentFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Preprocessed'
    cfg = RenderingCfg()

    cfg.sigma = 1e-9
    cfg.blurRange = 1e-9

    cfg.numCams = 16
    cfg.learningRate = 1e-4
    cfg.batchSize = 4
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
        str(i).zfill(5) for i in range(8564, 8708)
        # str(i).zfill(5) for i in range(8564, 8570)
    ]

    # Copy meshes generate by only lbs fitting and render it
    outFolderLBS = join(inputs.outFolder, 'PureLBS', )
    outFolderLBSMeshes = join(inputs.outFolder, 'PureLBS', 'Meshes')
    outFolderLBSImgs = join(inputs.outFolder, 'PureLBS', 'Rendered')
    os.makedirs(outFolderLBSMeshes, exist_ok=True)
    os.makedirs(outFolderLBSImgs, exist_ok=True)
    inFrameFolders = [join(inputs.inFolder, frameName) for frameName in frameNames]
    for frameName, inFrameFolder in zip(frameNames, inFrameFolders):
        pureLBSMesh = join(inFrameFolder, 'ToSparseMesh.obj')

        shutil.copy(pureLBSMesh, join(outFolderLBSMeshes, 'A' + frameName + '.obj'))

    renderConsecutiveFrames(outFolderLBSMeshes, inputs.cleanPlateFolder, inputs.texturedMesh, inputs.camParamF, outFolderLBSImgs, frameNames=frameNames, cfg=cfg, inMeshExt='obj', convertToM=True)

    # # Copy meshes generate by only lbs fitting and render it
    outFolderInterpolated = join(inputs.outFolder, 'Interpolated', )
    outFolderInterpolatedMeshes = join(inputs.outFolder, 'Interpolated', 'Meshes')
    outFolderInterpolatedImgs = join(inputs.outFolder, 'Interpolated', 'Rendered')
    os.makedirs(outFolderInterpolatedMeshes, exist_ok=True)
    os.makedirs(outFolderInterpolatedImgs, exist_ok=True)
    inFrameFolders = [join(inputs.inFolder, frameName) for frameName in frameNames]
    for frameName, inFrameFolder in zip(frameNames, inFrameFolders):
        pureLBSMesh = join(inFrameFolder, 'InterpolatedMesh.obj')

        shutil.copy(pureLBSMesh, join(outFolderInterpolatedMeshes, 'A' + frameName + '.obj'))

    renderConsecutiveFrames(outFolderInterpolatedMeshes, inputs.cleanPlateFolder, inputs.texturedMesh, inputs.camParamF, outFolderInterpolatedImgs, frameNames=frameNames, cfg=cfg, inMeshExt='obj', convertToM=True)