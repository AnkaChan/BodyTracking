from S04_GenerateTexture import *

class InputBundle():
    def __init__(s):

        s.camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
        s.smplshExampleMeshFile = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\SMPLSH.obj'
        s.toSparsePCMat = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\InterpolationMatrix.npy'
        s.inputDensePointCloudFile = None
        s.smplshData = r'..\Data\BuildSmplsh\Output\SmplshModel_m.npz'
        s.handIndicesFile = r'HandIndices.json'
        s.HeadIndicesFile = r'HeadIndices.json'
        s.personalShapeFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\PersonalShape.npy'
        s.betaFile = r'Z:\shareZ\2020_07_15_NewInitialFitting\InitialSilhouetteFitting\3052\Final\BetaFile.npy'
        s.texturedMesh = "..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj"
        s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.cleanPlateFolder = r'Z:\shareZ\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
        s.compressedStorage = True

        # frame specific inputs
        s.finalMeshFolder = None
        # s.initialFittingParamFile = None
        # s.cleanPlateFolder = None
        s.undistImgsFolder = None
        s.outFolder = None

if __name__ == '__main__':
    inputs = InputBundle()
    cfg = Config()

    inputs.finalMeshFolder = r'Z:\shareZ\2020_07_15_NewInitialFitting\CompleteTexture2WithSilhouette\Meshes'
    # s.initialFittingParamFile = r''
    inputs.outFolder = r'Z:\shareZ\2020_07_15_NewInitialFitting\CompleteTexture2WithSilhouette\output'
    inputs.undistImgsFolder = r'Z:\shareZ\2020_07_15_NewInitialFitting\CompleteTexture\UndistImgs'
    frameNames = ['03052', '03067', '04735', '06250', '06550']
    generateTexture(frameNames, inputs, cfg)