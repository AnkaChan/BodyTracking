from  M07_TextureGeneration import *
from os.path import join


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
        s.texturedMesh = r"..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj"
        s.compressedStorage = True

        s.finalMeshFolder = None
        s.undistImgsFolder = None
        s.outFolder = None

def generateTexture(frameNames, inputs=InputBundle(), cfg=Config()):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    contourOutFolder = join(inputs.outFolder, 'Size%dx%d' % (cfg.erosionSize, cfg.erosionSize))
    generateContourMask(inputs.camParamF, inputs.finalMeshFolder, frameNames, inputs.cleanPlateFolder, device, contourOutFolder, cfg)

    learnTexture(inputs.camParamF, inputs.finalMeshFolder, frameNames, device, inputs.undistImgsFolder, inputs.cleanPlateFolder, contourOutFolder, inputs.outFolder, cfg)

if __name__ == '__main__':
    inputs = InputBundle()

