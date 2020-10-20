import sys
sys.path.append('../AC_FitPipeline')
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Utility_Rendering import *
import Config
import M05_Visualization
from Utility import *
import tqdm

if __name__ == '__main__':
    cleanPlateFolder =  r'F:\WorkingCopy2\2020_07_26_NewPipelineTestData\CleanPlateExtracted\RgbUndist'
    inTextureMeshFile = r"..\Data\TextureMap2Color\SMPLWithSocks_tri.obj"
    camParamF = r'F:\WorkingCopy2\2020_05_31_DifferentiableRendererRealData\CameraParams\cam_params.json'

    outFolder = r'X:\MocapProj\2020_08_24_TexturedFitting_Lada_Rendering\OpticalFlowEvaluation'

    cfg = Config.RenderingCfg()
    cfg.sigma = 1e-10
    cfg.blurRange = 1e-10
    cfg.imgSize = 1080
    cfg.faces_per_pixel = 2

    cfg.numCams = 16
    cfg.batchSize = 8
    cfg.extrinsicsOutsideCamera = True

    fittingFolder = r'F:\WorkingCopy2\2020_07_28_TexturedFitting_Lada\Evaluation'

    ToKpAndDensefolder = join(fittingFolder, 'ToDense')
    ToTrackingPointsFolder = join(fittingFolder, 'ToTrackingPoints')
    InterpolatedFolder = join(fittingFolder, 'Interpolated')
    ImageBasedFittingFolder = join(fittingFolder, 'ImageBasedFitting')

    frames = [str(i) for i in range(10459 , 10459 + 300)]

    folders = [ToKpAndDensefolder, ToTrackingPointsFolder, InterpolatedFolder, ImageBasedFittingFolder]
    convertToMs = [ True,True, True, False]
    exts = ['obj', 'obj', 'obj', 'ply']

    for frameName in tqdm.tqdm(frames):
        for processFolder, convertToM, ext in zip(folders, convertToMs, exts):

            processRenderFolder = join(outFolder, os.path.basename(processFolder), frameName)
            meshFile = join(processFolder,  'A' + frameName + '.' + ext)
            os.makedirs(processRenderFolder, exist_ok=True)
            M05_Visualization.renderFrame(meshFile, inTextureMeshFile, camParamF,
                                                      processRenderFolder, cleanPlateFolder=cleanPlateFolder,
                                                      cfg=cfg, convertToM=convertToM, rendererType='RGB')