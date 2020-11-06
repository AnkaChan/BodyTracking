# convert to
# Undist images
# Reconstruct keypoints
# Fit to sparse point cloud and keypoint
# Interpolate using sparse point cloud

from S01_ToSparseFittingSelectedFrames import *

class InputBundle():
    def __init__(s):
        s.SMPLSHNpzFile = r'..\Data\BuildSmplsh_Female\Output\SmplshModel_f_noBun.npz'
        s.skelDataFile = r'..\Data\KateyBodyModel\InitialRegistration\06_SKelDataKeteyWeightsMultiplierCorrectAnkle_1692.json'

        s.inputDensePointCloudFile = None
        s.toSparsePCMat = r'..\Data\KateyBodyModel\InterpolationMatrix.npy'
        s.personalShapeFile = r'..\Data\KateyBodyModel\PersonalShape.npy'
        s.betaFile = r'..\Data\KateyBodyModel\beta.npy'
        s.camParamF = r'Z:\2020_01_01_KateyCapture\CameraParameters2_k1k2k3p1p2\cam_params.json'

        s.dataFolder = None
        s.deformedSparseMeshFolder = None
        s.inputKpFolder = None
        s.outFolderAll = None
        s.laplacianMatFile = None

if __name__ == '__main__':
    inputs = InputBundle()

    # # params for LadaStand
    # inputs.dataFolder = r'Z:\shareZ\2019_12_13_Lada_Capture\Converted'
    # inputs.deformedSparseMeshFolder = r'Z:\shareZ\2020_07_28_TexturedFitting_Lada\LadaStand'
    # inputs.outFolderAll = r'Z:\shareZ\2020_07_28_TexturedFitting_Lada'
    # inputs.preprocessOutFolder = r'Z:\shareZ\2020_07_28_TexturedFitting_Lada'
    # inputs.laplacianMatFile = r'SmplshRestposeLapMat.npy'
    #
    # camParamF = r'Z:\shareZ\2020_06_07_AC_ToSilhouetteFitting\CameraParams\cam_params.json'
    # # frameNames = [str(i).zfill(5) for i in range(8274, 10873)]
    # frameNames = [str(i).zfill(5) for i in range(9330, 10873)]
    # # frameNames = [str(i).zfill(5) for i in range(9745, 10874)]
    # # frameNames = [str(i).zfill(5) for i in range(8646 + 464, 10873)]
    # cfg = Config()
    # cfg.toSparseFittingCfg.learnrate_ph = 0.05
    # cfg.toSparseFittingCfg.lrDecayStep = 200
    # cfg.toSparseFittingCfg.lrDecayRate = 0.96
    # cfg.toSparseFittingCfg.numIterFitting = 6000
    # cfg.toSparseFittingCfg.noBodyKeyJoint = True
    # cfg.toSparseFittingCfg.betaRegularizerWeightToKP = 1000
    # cfg.toSparseFittingCfg.outputErrs = True
    # cfg.toSparseFittingCfg.terminateLossStep = 1e-8
    # cfg.toSparseFittingCfg.

    # params for KeteyLongSeq
    inputs.dataFolder = r'Z:\2020_01_01_KateyCapture\Converted'
    # inputs.deformedSparseMeshFolder = r'Z:\2020_08_26_TexturedFitting_LadaGround\LadaGround'
    inputs.deformedSparseMeshFolder = r'Z:\2020_08_27_KateyBodyModel\Deformed_Weight1'
    # inputs.outFolderAll = r'Z:\2020_08_27_KateyBodyModel\TPose'
    inputs.outFolderAll = r'Z:\2020_08_27_KateyBodyModel\All'
    # inputs.preprocessOutFolder = r'Z:\shareZ\2020_08_27_KateyBodyModel\TPose'

    # inputs.outFolderAll = r'Z:\2020_08_27_KateyBodyModel\Backbend'
    # inputs.preprocessOutFolder = r'Z:\2020_08_27_KateyBodyModel\Backbend'

    inputs.preprocessOutFolder = inputs.outFolderAll
    inputs.laplacianMatFile = r'SmplshRestposeLapMat_Katey.npy'

    inputs.camParamF = r'Z:\2020_01_01_KateyCapture\CameraParameters2_k1k2k3p1p2\cam_params.json'
    # frameNames = [str(i).zfill(5) for i in range(8274, 10873)]
    # frameNames = [str(i).zfill(5) for i in range(14946 , 17745)]
    # frameNames = [str(i).zfill(5) for i in range(16270 , 17745)]
    frameNames = [str(i).zfill(5) for i in range(16659 , 10873)]

    # frameNames = [str(i).zfill(5) for i in range(18410 , 18414)]
    # frameNames = ['16755']
    # frameNames = ['16755']
    # frameNames = [str(i).zfill(5) for i in range(9745, 10874)]
    # frameNames = [str(i).zfill(5) for i in range(8646 + 464, 10873)]
    cfg = Config()
    cfg.toSparseFittingCfg.learnrate_ph = 0.05
    cfg.toSparseFittingCfg.lrDecayStep = 200
    cfg.toSparseFittingCfg.lrDecayRate = 0.96
    cfg.toSparseFittingCfg.numIterFitting = 6000
    cfg.toSparseFittingCfg.noBodyKeyJoint = True
    cfg.toSparseFittingCfg.betaRegularizerWeightToKP = 1000
    cfg.toSparseFittingCfg.outputErrs = True
    cfg.toSparseFittingCfg.terminateLossStep = 1e-8
    cfg.toSparseFittingCfg.withFaceKp = True


    cfg.kpReconCfg.openposeModelDir = r"C:\Code\Project\Openpose\models"
    cfg.kpReconCfg.numMostConfidentToPick =2
    # cfg.kpReconCfg.debugFolder =
    # cfg.kpReconCfg.drawResults = False
    cfg.kpReconCfg.drawResults = True
    cfg.kpReconCfg.detecHead = True
    cfg.kpReconCfg.rescale = True
    cfg.kpReconCfg.reprojectErrThreshold = 30
    # cfg.kpReconCfg.openposeModelDir = r"Z:\Anka\OpenPose\models"

    # preprocess
    preprocessSelectedFrame(inputs.dataFolder, frameNames, inputs.camParamF, inputs.preprocessOutFolder, cfg)

    # to sparse fitting

    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    inputs.inputKpFolder = join(inputs.outFolderAll, 'Keypoints')
    # toSparseFittingSelectedFrame(inputs, frameNames, cfg)

    # intepolate to sparse mesh
    # interpolateToSparseMeshSelectedFrame(inputs, frameNames)

"""Preprocessing:  21%|██        | 464/2227 [1:56:13<7:21:35, 15.03s/it]
Traceback (most recent call last):
  File "Z:/Anka/BodyTracking2/BodyTracking/AC_FitPipeline/S01_ToSparseFittingSelectedFrames_DataChewer.py", line 50, in <module>
    preprocessSelectedFrame(inputs.dataFolder, frameNames, inputs.camParamF, inputs.preprocessOutFolder, cfg)
  File "Z:\Anka\BodyTracking2\BodyTracking\AC_FitPipeline\S01_ToSparseFittingSelectedFrames.py", line 50, in preprocessSelectedFrame
    M02_ReconstructionJointFromRealImagesMultiFolder.reconstructKeypoints2(rgbUndistFrameFiles, outKpFile, camParamF, cfg.kpReconCfg, )
  File "Z:\Anka\BodyTracking2\BodyTracking\AC_FitPipeline\M02_ReconstructionJointFromRealImagesMultiFolder.py", line 549, in reconstructKeypoints2
    if len(handKeyPointsRight) and handKeyPointsRight.shape[0] == numHandpts:
TypeError: len() of unsized object"""