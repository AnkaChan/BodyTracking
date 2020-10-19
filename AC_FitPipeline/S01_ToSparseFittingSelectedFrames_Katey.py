# convert to
# Undist images
# Reconstruct keypoints
# Fit to sparse point cloud and keypoint
# Interpolate using sparse point cloud

from S01_ToSparseFittingSelectedFrames import *

class InputBundle():
    def __init__(s):
        s.SMPLSHNpzFile = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.skelDataFile = r'C:\Code\MyRepo\ChbCapture\06_Deformation\MeshInterpolation\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'

        # s.inputImgDataFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied'
        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'F:\WorkingCopy2\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'

        s.dataFolder = None
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = None
        s.outFolderAll = None
        s.laplacianMatFile = None


if __name__ == '__main__':
    inputs = InputBundle()

    # inputs.dataFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.preprocessOutFolder = r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData'
    # inputs.deformedSparseMeshFolder = r''
    # inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\ObjFiles'
    # inputs.inputKpFolder = join(inputs.dataFolder, 'Keypoints')
    # inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inputs.outFolderAll = inputs.dataFolder
    # frameNames = [
    # #              '03067',
    #               # '03990',
    #               # '04735', '04917',
    #               # '06250',
    #               '06550',
    #               #  '06950'
    #               ]

    inputs.dataFolder = r'Z:\2020_01_01_KateyCapture\Converted'
    # inputs.preprocessOutFolder = r'Z:\2020_08_27_KateyBodyModel\TPose'
    inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_03_19_Katey_WholeSeq\TPose2\SLap_SBiLap_True_TLap_0_JTW_5000_JBiLap_0_Step1_Overlap0\Deformed'
    inputs.inputKpFolder = join(inputs.dataFolder, 'Keypoints')
    inputs.camParamF = r'Z:\2020_01_01_KateyCapture\CameraParameters3_k6p2\cam_params.json'
    # inputs.camParamF = r'Z:\2020_01_01_KateyCapture\CameraParameters\cam_params.json'
    # inputs.outFolderAll = inputs.dataFolder
    inputs.preprocessOutFolder = r'Z:\2020_09_10_CleanPlateKatey'

    # frameNames = [str(iFrame).zfill(5) for iFrame in  range(18410, 18414)]
    # Clean Plate
    frameNames = [str(iFrame).zfill(5) for iFrame in range(3280, 3281)]

    # inputs.dataFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada'
    # inputs.outFolderAll = inputs.dataFolder
    # inputs.deformedSparseMeshFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\LadaStand'
    # inputs.camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'
    # inputs.inputKpFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Keypoints'

    # frameNames = [str(iFrame).zfill(5) for iFrame in range(8332, 8332 + 5)]

    cfg = Config()
    cfg.saveDistRgb = True
    cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.learnrate_ph = 0.005
    cfg.toSparseFittingCfg.lrDecayStep = 200
    cfg.toSparseFittingCfg.lrDecayRate = 0.96
    cfg.toSparseFittingCfg.numIterFitting = 6000
    cfg.toSparseFittingCfg.noBodyKeyJoint = True
    cfg.toSparseFittingCfg.betaRegularizerWeightToKP = 1000
    cfg.toSparseFittingCfg.outputErrs = True

    cfg.kpReconCfg.openposeModelDir = r"C:\Code\Project\Openpose\models"
    # cfg.kpReconCfg.drawResults = True
    cfg.kpReconCfg.debugFolder = join(inputs.preprocessOutFolder, 'Preprocessed', 'Debug')

    # camFolders = sortedGlob(join(dataFolder, '*'))
    # imgFolders = sortedGlob(r'F:\WorkingCopy2\2020_05_21_AC_FramesDataToFitTo\Copied\*')
    #
    # for imgFolder in imgFolders[:-3]:
    #     imgFs = sortedGlob(join(imgFolder, '*.pgm'))
    #
    #     for imgF, camFolder in zip(imgFs, camFolders):
    #         shutil.copy(imgF, join(camFolder, os.path.basename(imgF)))
    # # preprocess
    preprocessSelectedFrame(inputs.dataFolder, frameNames, inputs.camParamF, inputs.preprocessOutFolder, cfg)

    # to sparse fitting

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    toSparseFittingSelectedFrame(inputs, frameNames, cfg)

    # intepolate to sparse mesh
    interpolateToSparseMeshSelectedFrame(inputs, frameNames)