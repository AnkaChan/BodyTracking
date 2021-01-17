import sys
sys.path.append('../AC_FitPipeline')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Utility_Rendering import *
import Config
import M05_Visualization

if __name__ == '__main__':
    # renderConsecutiveFrames(inFramesFolder, cleanPlateFolder, inTextureMeshFile, camParamF, outFolder,
    #                             frameNames=None, cfg=RenderingCfg(),
    #                             inMeshExt='ply', convertToM=False, rendererType='RGB')
    # inFramesFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ToKpOnly\All'
    # inFramesFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\Interpolated'
    # inFramesFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation\ImageBasedFitting'
    # camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    inFramesFolder = r'C:\Code\MyRepo\03_capture\Mocap-CVPR-Paper-Figures\04_SIlhouetteOptimization\Data\Mesh'
    camParamF = r'F:\WorkingCopy2\2020_01_01_KateyCapture\CameraParameters3_k6p2\cam_params.json'
    convertToM =True
    # ext = 'obj'
    ext = 'ply'

    outFolder = os.path.join(inFramesFolder, r'Silhouettes')


    cleanPlateFolder =  r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    inTextureMeshFile = r"..\Data\TextureMap2Color\Initial1Frame\SMPLWithSocks_tri.obj"

    cfg = Config.RenderingCfg()
    cfg.sigma = 1e-10
    cfg.blurRange = 1e-10
    cfg.imgSize = 1080
    cfg.faces_per_pixel = 5

    cfg.numCams = 16
    cfg.batchSize = 8

    os.makedirs(outFolder, exist_ok=True)
    M05_Visualization.renderConsecutiveFrames(inFramesFolder, cleanPlateFolder, inTextureMeshFile, camParamF, outFolder,
                                frameNames=None, cfg=cfg,
                                inMeshExt=ext, convertToM=convertToM, rendererType='Silhouette')

