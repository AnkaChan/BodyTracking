from S23_Pipeline_IntialToSilhouetteFitting import *

class InputBundle:
    def __init__(s):
        # same over all frames
        s.camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\SMPLSH.obj'
        s.toSparsePCMat = r'Z:\shareZ\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\InterpolationMatrix.npy'
        s.smplshRegressorMatFile = r'smplshRegressorNoFlatten.npy'
        # s.smplshData = r'..\SMPL_reimp\SmplshModel_m.npz'
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'

        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'Z:\shareZ\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'
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

from S05_InterpolateWithSparsePointCloud import interpolateWithSparsePointCloudSoftly


if __name__ == '__main__':
    ### This is the firsting fitting to silhouette, before we have it register to sparse point cloud
    # before we have the texture
    inputs = InputBundle()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    cfgPoseFitting = RenderingCfg()

    cfgPoseFitting.sigma = 1e-7
    cfgPoseFitting.blurRange = 1e-7
    # cfgPoseFitting.plotStep = 20
    cfgPoseFitting.plotStep = 100
    cfgPoseFitting.numCams = 16
    # low learning rate for pose optimization
    cfgPoseFitting.learningRate = 2e-5
    cfgPoseFitting.batchSize = 4
    # cfgPoseFitting.faces_per_pixel = 6 # for testing
    cfgPoseFitting.faces_per_pixel = 6 # for debugging
    # cfgPoseFitting.imgSize = 2160
    cfgPoseFitting.imgSize = 1080
    cfgPoseFitting.terminateLoss = 0.1
    cfgPoseFitting.lpSmootherW = 0.000001
    # cfgPoseFitting.normalSmootherW = 0.1
    cfgPoseFitting.normalSmootherW = 0.0
    cfgPoseFitting.numIterations = 2000
    # cfgPoseFitting.numIterations = 20
    cfgPoseFitting.kpFixingWeight = 0.01
    cfgPoseFitting.bin_size = 256

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
    cfgPerVert.lpSmootherW = 1e-7
    cfgPerVert.vertexFixingWeight = 0
    cfgPerVert.normalSmootherW = 0.0
    cfgPerVert.numIterations = 500
    # cfgPerVert.numIterations = 20
    cfgPerVert.bin_size = 256

    # frameName = '3052'
    # undistortSilhouette = False

    frameName = '06250'
    undistortSilhouette = True

    inputs.imageFolder = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\\' + frameName + r'\Silhouettes'
    # inputs.outputFolder = join(r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\Output', frameName)
    inputs.outputFolder = join(r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting', frameName)
    inputs.finalOutputFolder = join(r'Z:\shareZ\2020_07_15_NewInitialFitting\Final', frameName)

    # inputs.compressedStorage = False
    # inputs.initialFittingParamPoseFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialRegistration\OptimizedPoses_ICPTriangle.npy'
    # inputs.initialFittingParamBetasFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialRegistration\OptimizedBetas_ICPTriangle.npy'
    # inputs.initialFittingParamTranslationFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialRegistration\OptimizedTranslation_ICPTriangle.npy'

    inputs.compressedStorage = True
    inputs.initialFittingParamFile = r'Z:\shareZ\2020_07_26_NewPipelineTestData\ToSparse\06250\ToSparseFittingParams.npz'

    inputs.KeypointsFile = r'Z:\shareZ\2020_06_14_FitToMultipleCams\KepPoints\\'+frameName+'.obj'

    inputsPose = copy(inputs)
    inputsPose.outputFolder = join(inputs.outputFolder, 'SilhouettePose')
    toSilhouettePoseInitalFitting(inputsPose, cfgPoseFitting, device, undistortSilhouettes=undistortSilhouette)
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
    inputsPerVertFitting.initialFittingParamFile = finalPoseFile
    toSilhouettePerVertexInitialFitting(inputsPerVertFitting, cfgPerVert, device)
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






