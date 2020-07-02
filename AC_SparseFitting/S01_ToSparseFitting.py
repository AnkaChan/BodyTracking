# import M01_ReconstructionJointFromRealImagesMultiFolder
import M03_ToSparseFitting

import glob, os, json
from os.path import join
import numpy as np

class InputBundle:
    def __init__(s):
        # person specific
        s.skelDataFile = r'..\Data\PersonalModel_Lada\06_SKelDataLadaWeightsMultiplierCorrectAnkle_1692.json'
        s.toSparsePointCloudInterpoMatFile = r'..\Data\PersonalModel_Lada\InterpolationMatrix.npy'

        s.betaFile = r'..\Data\PersonalModel_Lada\Beta.npy'
        s.personalShapeFile = r'F:\WorkingCopy2\2020_06_14_FitToMultipleCams\InitialFit\PersonalModel\PersonalShape.npy'

        s.OP2AdamJointMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\OP2AdamJointMat.npy'
        s.AdamGoodJointsFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\AdamGoodJoints.npy'
        s.smplsh2OPRegressorMatFile = r'..\Data\PersonalModel_Lada\OpenposeToSmplsh\smplshRegressorNoFlatten.npy'
        s.smplshDataFile = r'..\SMPL_reimp\SmplshModel_m.npz'


class Config:
    def __init__(s):
        # s.keypointsDetectionCfg = M01_ReconstructionJointFromRealImagesMultiFolder.Config()
        s.toSparseFittignCfg = M03_ToSparseFitting.Config()
        pass

if __name__ == '__main__':
    inputs = InputBundle()
    inImgParentFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\Copied\Images'
    camParamFile = r'C:\Code\MyRepo\03_capture\BodyTracking\Data\CamParams\Lada_19_12_13\cam_params.json'
    completedObjFolder=r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\Copied\Deformed\SLap_SBiLap_True_TLap_0_JTW_0_JBiLap_0_Step10_Overlap0\Deformed'
    outFolder = r'F:\WorkingCopy2\2020_06_30_AC_ConsequtiveTexturedFitting\ToSparse'
    cfg = Config()

    inImgFolders = glob.glob(join(inImgParentFolder, '*'))
    inObjFiles = glob.glob(join(completedObjFolder, '*.obj'))
    inImgFolders.sort()
    inObjFiles.sort()

    # # openpose key points detection
    # for inFolder in inImgFolders:
    #     M01_ReconstructionJointFromRealImagesMultiFolder.reconstructKeypoints(inFolder, camParamFile, cfg.keypointsDetectionCfg)

    # complete the sparse point cloud
    # fitParam = np.load(r'..\Data\PersonalModel_Lada\FittingParam.npz')
    # betas = fitParam['beta']
    # np.save(inputs.betaFile, betas)

    for inImgFolder, objFile in zip(inImgFolders, inObjFiles):
        frameName = os.path.basename(inImgFolder)
        outFolderFrame = join(outFolder, join(outFolder, frameName))
        os.makedirs(outFolderFrame, exist_ok=True)
        M03_ToSparseFitting.toSparseFitting(inImgFolder, objFile, outFolderFrame, inputs.skelDataFile, inputs.toSparsePointCloudInterpoMatFile,
                        inputs.betaFile, inputs.personalShapeFile, inputs.OP2AdamJointMatFile, inputs.AdamGoodJointsFile, inputs.smplsh2OPRegressorMatFile,
                        smplshDataFile=inputs.smplshDataFile)